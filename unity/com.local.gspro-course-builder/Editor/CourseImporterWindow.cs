using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace GSProCourseBuilder.Editor
{
    /// <summary>
    /// GSPro Builder > Import Course Output...
    /// Builds a Unity 6 scene from a gspro_builder output folder:
    ///   terrain/terrain.json + heightmap_N.raw  -> Terrain (real-world metres, centred on 0,0,0)
    ///   surfaces/splat_control0/1.png           -> 8 terrain layers (rough..cartpath)
    ///   overlay.jpg                             -> optional "aerial" layer draped over the terrain
    ///   holes.json                              -> Hole01.. objects with tee/pin/aim markers
    /// This is a preview/design environment. The GSPro build itself runs in the
    /// OPCD Unity project (see docs/PIPELINE.md).
    /// </summary>
    public class CourseImporterWindow : EditorWindow
    {
        string outputDir = "";
        bool useOverlay = false;
        bool importHoles = true;
        float layerTileSize = 4f;

        static readonly Dictionary<string, Color> LayerColors = new Dictionary<string, Color>
        {
            { "rough", new Color32(39, 132, 56, 255) }, { "trees", new Color32(22, 75, 32, 255) },
            { "fairway", new Color32(67, 229, 97, 255) }, { "tee", new Color32(160, 229, 184, 255) },
            { "green", new Color32(188, 229, 164, 255) }, { "bunker", new Color32(229, 229, 170, 255) },
            { "water", new Color32(0, 0, 192, 255) }, { "cartpath", new Color32(128, 128, 128, 255) },
        };

        [MenuItem("GSPro Builder/Import Course Output...")]
        static void Open() => GetWindow<CourseImporterWindow>("GSPro Course Import");

        void OnGUI()
        {
            EditorGUILayout.LabelField("gspro_builder output folder", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                outputDir = EditorGUILayout.TextField(outputDir);
                if (GUILayout.Button("Browse", GUILayout.Width(70)))
                {
                    var p = EditorUtility.OpenFolderPanel("courses/<slug>/output", outputDir, "");
                    if (!string.IsNullOrEmpty(p)) outputDir = p;
                }
            }
            useOverlay = EditorGUILayout.Toggle("Drape aerial overlay", useOverlay);
            importHoles = EditorGUILayout.Toggle("Create hole markers", importHoles);
            layerTileSize = EditorGUILayout.FloatField("Surface texture tile (m)", layerTileSize);
            EditorGUILayout.HelpBox("Surface layers get flat placeholder colours. Assign real grass/sand " +
                                    "textures to the generated TerrainLayer assets afterwards.", MessageType.Info);
            GUI.enabled = Directory.Exists(outputDir);
            if (GUILayout.Button("Import", GUILayout.Height(30))) Import(outputDir, useOverlay, importHoles, layerTileSize);
            GUI.enabled = true;
        }

        public static void Import(string dir, bool overlay, bool holes, float tile)
        {
            var tjPath = Path.Combine(dir, "terrain", "terrain.json");
            if (!File.Exists(tjPath)) { EditorUtility.DisplayDialog("GSPro Builder", "terrain/terrain.json not found", "OK"); return; }
            var tj = JsonUtility.FromJson<TerrainJson>(File.ReadAllText(tjPath));
            var slug = new DirectoryInfo(dir).Parent?.Name ?? "course";
            var assetDir = $"Assets/GSProCourses/{slug}";
            Directory.CreateDirectory(assetDir);

            try
            {
                EditorUtility.DisplayProgressBar("GSPro Builder", "Heightmap", 0.1f);
                var td = new TerrainData { heightmapResolution = tj.resolution };
                td.size = new Vector3(tj.size_m, tj.height_range_m, tj.size_m);
                td.SetHeights(0, 0, ReadRaw(Path.Combine(dir, "terrain", tj.raw_file), tj.resolution, tj.unity_flip_vertically));

                EditorUtility.DisplayProgressBar("GSPro Builder", "Surface layers", 0.4f);
                ApplySurfaces(td, dir, assetDir, overlay, tile, tj.size_m);

                var tdPath = $"{assetDir}/{slug}_TerrainData.asset";
                AssetDatabase.DeleteAsset(tdPath);
                AssetDatabase.CreateAsset(td, tdPath);

                var old = GameObject.Find($"Terrain_{slug}");
                if (old) Undo.DestroyObjectImmediate(old);
                var go = Terrain.CreateTerrainGameObject(td);
                go.name = $"Terrain_{slug}";
                go.transform.position = new Vector3(-tj.size_m / 2f, 0f, -tj.size_m / 2f);
                var terrain = go.GetComponent<Terrain>();
                terrain.heightmapPixelError = 2f;
                terrain.basemapDistance = 2000f;
                Undo.RegisterCreatedObjectUndo(go, "Import GSPro course");

                if (holes)
                {
                    EditorUtility.DisplayProgressBar("GSPro Builder", "Holes", 0.8f);
                    ImportHoles(Path.Combine(dir, "holes.json"), terrain, slug);
                }
                AssetDatabase.SaveAssets();
                Debug.Log($"[GSPro Builder] Imported {slug}: {tj.size_m} m terrain, elevation " +
                          $"{tj.min_elev_m:F1}-{tj.max_elev_m:F1} m (terrain y=0 is {tj.min_elev_m:F1} m above sea level).");
            }
            finally { EditorUtility.ClearProgressBar(); }
        }

        static float[,] ReadRaw(string path, int n, bool flip)
        {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length != n * n * 2) throw new IOException($"{path}: expected {n * n * 2} bytes, got {bytes.Length}");
            var h = new float[n, n];
            for (int row = 0; row < n; row++)
            {
                int z = flip ? n - 1 - row : row;
                for (int x = 0; x < n; x++)
                {
                    int i = (row * n + x) * 2;
                    h[z, x] = (bytes[i] | (bytes[i + 1] << 8)) / 65535f;
                }
            }
            return h;
        }

        static Color32[] LoadPng(string path, out int w, out int h)
        {
            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            tex.LoadImage(File.ReadAllBytes(path));   // rows come back bottom-up = south-first
            w = tex.width; h = tex.height;
            var px = tex.GetPixels32();
            DestroyImmediate(tex);
            return px;
        }

        static void ApplySurfaces(TerrainData td, string dir, string assetDir, bool overlay, float tile, float size)
        {
            var sjPath = Path.Combine(dir, "surfaces", "surfaces.json");
            var sj = JsonUtility.FromJson<SurfacesJson>(File.ReadAllText(sjPath));
            var layers = new List<TerrainLayer>();
            foreach (var l in sj.layers.OrderBy(l => l.index))
                layers.Add(MakeLayer(assetDir, l.name, LayerColors.TryGetValue(l.name, out var c) ? c : Color.magenta, tile));

            TerrainLayer aerial = null;
            var ovSrc = Path.Combine(dir, "overlay.jpg");
            if (overlay && File.Exists(ovSrc))
            {
                var ovDst = $"{assetDir}/overlay.jpg";
                File.Copy(ovSrc, ovDst, true);
                AssetDatabase.ImportAsset(ovDst);
                var imp = (TextureImporter)AssetImporter.GetAtPath(ovDst);
                imp.maxTextureSize = 8192; imp.wrapMode = TextureWrapMode.Clamp; imp.mipmapEnabled = true;
                imp.SaveAndReimport();
                aerial = MakeLayer(assetDir, "aerial", Color.white, size, AssetDatabase.LoadAssetAtPath<Texture2D>(ovDst));
                layers.Add(aerial);
            }
            td.terrainLayers = layers.ToArray();

            int res = Mathf.ClosestPowerOfTwo(Mathf.Clamp(sj.resolution, 16, 4096));
            td.alphamapResolution = res;
            res = td.alphamapResolution;
            var maps = new float[res, res, layers.Count];
            if (aerial != null)
            {
                int ai = layers.Count - 1;
                for (int y = 0; y < res; y++) for (int x = 0; x < res; x++) maps[y, x, ai] = 1f;
            }
            else
            {
                for (int k = 0; k < 2; k++)
                {
                    var px = LoadPng(Path.Combine(dir, "surfaces", $"splat_control{k}.png"), out int w, out int h);
                    for (int y = 0; y < res; y++)
                    {
                        int sy = y * h / res;
                        for (int x = 0; x < res; x++)
                        {
                            var c = px[sy * w + x * w / res];
                            int b = 4 * k;
                            maps[y, x, b + 0] = c.r / 255f;
                            maps[y, x, b + 1] = c.g / 255f;
                            maps[y, x, b + 2] = c.b / 255f;
                            maps[y, x, b + 3] = c.a / 255f;
                        }
                    }
                }
            }
            td.SetAlphamaps(0, 0, maps);
        }

        static TerrainLayer MakeLayer(string assetDir, string name, Color color, float tile, Texture2D tex = null)
        {
            if (tex == null)
            {
                var texPath = $"{assetDir}/Layer_{name}.asset";
                tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
                if (tex == null)
                {
                    tex = new Texture2D(4, 4, TextureFormat.RGBA32, true) { name = $"Layer_{name}" };
                    tex.SetPixels(Enumerable.Repeat(color, 16).ToArray());
                    tex.Apply();
                    AssetDatabase.CreateAsset(tex, texPath);
                }
            }
            var path = $"{assetDir}/{name}.terrainlayer";
            var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(path);
            if (layer == null)   // keep user-assigned textures on re-import
            {
                layer = new TerrainLayer { diffuseTexture = tex, tileSize = new Vector2(tile, tile) };
                AssetDatabase.CreateAsset(layer, path);
            }
            else if (name == "aerial") { layer.diffuseTexture = tex; layer.tileSize = new Vector2(tile, tile); }
            return layer;
        }

        static void ImportHoles(string path, Terrain terrain, string slug)
        {
            if (!File.Exists(path)) { Debug.LogWarning("[GSPro Builder] holes.json not found"); return; }
            var hj = JsonUtility.FromJson<HolesJson>(File.ReadAllText(path));
            var rootName = $"Holes_{slug}";
            var old = GameObject.Find(rootName);
            if (old) Undo.DestroyObjectImmediate(old);
            var root = new GameObject(rootName);
            Undo.RegisterCreatedObjectUndo(root, "Import holes");
            foreach (var h in hj.holes)
            {
                var hole = new GameObject($"Hole{h.number:00}");
                hole.transform.SetParent(root.transform, false);
                var route = hole.AddComponent<HoleRoute>();
                route.number = h.number; route.par = h.par; route.handicap = h.handicap;
                foreach (var t in h.tees)
                    Marker(hole, $"Tee_{t.name}", MarkerKind.Tee, h.number, t.position, terrain).teeName = t.name;
                Marker(hole, "Pin", MarkerKind.Pin, h.number, h.pin, terrain);
                for (int i = 0; i < h.aim_points.Length; i++)
                    Marker(hole, $"Aim_{i + 1}", MarkerKind.AimPoint, h.number, h.aim_points[i], terrain).order = i;
                Marker(hole, "GreenCenter", MarkerKind.GreenCenter, h.number, h.green_center, terrain);
            }
        }

        static HoleMarker Marker(GameObject parent, string name, MarkerKind kind, int hole, V3 p, Terrain terrain)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            var pos = p.ToVector3();
            pos.y = terrain.SampleHeight(pos) + terrain.transform.position.y;
            go.transform.position = pos;
            var m = go.AddComponent<HoleMarker>();
            m.hole = hole; m.kind = kind;
            return m;
        }
    }
}
