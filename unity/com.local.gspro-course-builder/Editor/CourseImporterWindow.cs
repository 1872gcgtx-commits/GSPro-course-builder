using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace GSProCourseBuilder.Editor
{
    /// <summary>
    /// GSPro Builder > Import Course Output...
    /// Builds a Unity 6 scene from a gspro_builder output folder:
    ///   terrain/terrain.json + heightmap_N.raw  -> Terrain (graded, real-world metres, centred on 0,0,0)
    ///   surfaces/surfaces.json + splat maps     -> terrain layers (rough, trees, first cut, fairway,
    ///                                              fringe, tee, green, bunker, water, cart path)
    ///   trees.csv                               -> terrain trees (your prefabs, or placeholders)
    ///   water.json                              -> flat water planes at the graded water level
    ///   course_surfaces.obj                     -> conformed Hole##_Green/Tee/Fairway/Bunker meshes
    ///   holes.json                              -> Hole01.. with tees, pin, pin positions, aim points
    ///   overlay.jpg                             -> optional aerial layer draped over the terrain
    /// Works in any Unity 6 render pipeline (Built-in, URP, HDRP), including inside the
    /// OPCD V4 Unity 6 project.
    /// </summary>
    public class CourseImporterWindow : EditorWindow
    {
        string outputDir = "";
        bool useOverlay, importHoles = true, importTrees = true, importWater = true, importMeshes;
        float layerTileSize = 4f;
        GameObject deciduousPrefab, coniferPrefab;
        Material waterMaterial;

        static readonly Dictionary<string, Color> LayerColors = new Dictionary<string, Color>
        {
            { "rough", new Color32(39, 132, 56, 255) }, { "trees", new Color32(22, 75, 32, 255) },
            { "first_cut", new Color32(52, 170, 74, 255) }, { "fairway", new Color32(67, 229, 97, 255) },
            { "fringe", new Color32(120, 214, 120, 255) }, { "tee", new Color32(160, 229, 184, 255) },
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
            EditorGUILayout.Space();
            useOverlay = EditorGUILayout.Toggle("Drape aerial overlay", useOverlay);
            layerTileSize = EditorGUILayout.FloatField("Surface texture tile (m)", layerTileSize);
            importHoles = EditorGUILayout.Toggle("Hole markers", importHoles);
            importWater = EditorGUILayout.Toggle("Water planes", importWater);
            if (importWater) waterMaterial = (Material)EditorGUILayout.ObjectField("  Water material", waterMaterial, typeof(Material), false);
            importTrees = EditorGUILayout.Toggle("Trees", importTrees);
            if (importTrees)
            {
                deciduousPrefab = (GameObject)EditorGUILayout.ObjectField("  Deciduous prefab", deciduousPrefab, typeof(GameObject), false);
                coniferPrefab = (GameObject)EditorGUILayout.ObjectField("  Conifer prefab", coniferPrefab, typeof(GameObject), false);
            }
            importMeshes = EditorGUILayout.Toggle("Course surface meshes (OBJ)", importMeshes);
            EditorGUILayout.HelpBox("Empty slots get placeholder assets (flat-colour layers, simple trees, " +
                                    "blue water). Swap in real grass/sand textures, SpeedTree/tree prefabs " +
                                    "and your water shader afterwards - re-importing keeps your layer textures.",
                                    MessageType.Info);
            GUI.enabled = Directory.Exists(outputDir);
            if (GUILayout.Button("Import", GUILayout.Height(30)))
                Import(new Options
                {
                    dir = outputDir, overlay = useOverlay, holes = importHoles, trees = importTrees,
                    water = importWater, meshes = importMeshes, tile = layerTileSize,
                    deciduous = deciduousPrefab, conifer = coniferPrefab, waterMat = waterMaterial,
                });
            GUI.enabled = true;
        }

        public class Options
        {
            public string dir;
            public bool overlay, holes = true, trees = true, water = true, meshes;
            public float tile = 4f;
            public GameObject deciduous, conifer;
            public Material waterMat;
        }

        public static void Import(Options o)
        {
            var tjPath = Path.Combine(o.dir, "terrain", "terrain.json");
            if (!File.Exists(tjPath)) { EditorUtility.DisplayDialog("GSPro Builder", "terrain/terrain.json not found", "OK"); return; }
            var tj = JsonUtility.FromJson<TerrainJson>(File.ReadAllText(tjPath));
            var slug = new DirectoryInfo(o.dir).Parent?.Name ?? "course";
            var assetDir = $"Assets/GSProCourses/{slug}";
            Directory.CreateDirectory(assetDir);

            try
            {
                EditorUtility.DisplayProgressBar("GSPro Builder", "Heightmap", 0.1f);
                var td = new TerrainData { heightmapResolution = tj.resolution };
                td.size = new Vector3(tj.size_m, tj.height_range_m, tj.size_m);
                td.SetHeights(0, 0, ReadRaw(Path.Combine(o.dir, "terrain", tj.raw_file), tj.resolution, tj.unity_flip_vertically));

                EditorUtility.DisplayProgressBar("GSPro Builder", "Surface layers", 0.3f);
                ApplySurfaces(td, o.dir, assetDir, o.overlay, o.tile, tj.size_m);

                var tdPath = $"{assetDir}/{slug}_TerrainData.asset";
                AssetDatabase.DeleteAsset(tdPath);
                AssetDatabase.CreateAsset(td, tdPath);

                var old = GameObject.Find($"Terrain_{slug}");
                if (old) Undo.DestroyObjectImmediate(old);
                var go = Terrain.CreateTerrainGameObject(td);
                go.name = $"Terrain_{slug}";
                go.transform.position = new Vector3(-tj.size_m / 2f, 0f, -tj.size_m / 2f);
                var terrain = go.GetComponent<Terrain>();
                terrain.heightmapPixelError = 1f;
                terrain.basemapDistance = 2000f;
                terrain.treeDistance = 2000f;
                terrain.treeBillboardDistance = 300f;
                Undo.RegisterCreatedObjectUndo(go, "Import GSPro course");

                var root = Root(slug);
                if (o.trees)
                {
                    EditorUtility.DisplayProgressBar("GSPro Builder", "Trees", 0.5f);
                    ImportTrees(Path.Combine(o.dir, "trees.csv"), terrain, assetDir, o.deciduous, o.conifer);
                }
                if (o.water)
                {
                    EditorUtility.DisplayProgressBar("GSPro Builder", "Water", 0.65f);
                    ImportWater(Path.Combine(o.dir, "water.json"), root, assetDir, o.waterMat);
                }
                if (o.meshes)
                {
                    EditorUtility.DisplayProgressBar("GSPro Builder", "Surface meshes", 0.75f);
                    ImportSurfaceMeshes(Path.Combine(o.dir, "course_surfaces.obj"), root, assetDir);
                }
                if (o.holes)
                {
                    EditorUtility.DisplayProgressBar("GSPro Builder", "Holes", 0.85f);
                    ImportHoles(Path.Combine(o.dir, "holes.json"), terrain, root);
                }
                AssetDatabase.SaveAssets();
                Debug.Log($"[GSPro Builder] Imported {slug}: {tj.size_m} m terrain, elevation " +
                          $"{tj.min_elev_m:F1}-{tj.max_elev_m:F1} m (terrain y=0 is {tj.min_elev_m:F1} m above sea level).");
            }
            finally { EditorUtility.ClearProgressBar(); }
        }

        static GameObject Root(string slug)
        {
            var name = $"Course_{slug}";
            var old = GameObject.Find(name);
            if (old) Undo.DestroyObjectImmediate(old);
            var root = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(root, "Import GSPro course");
            return root;
        }

        // ---------------------------------------------------------------- terrain
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

        static Material NewLitMaterial(Color c, string name)
        {
            var rp = GraphicsSettings.defaultRenderPipeline;
            var baseMat = rp != null ? rp.defaultMaterial : AssetDatabase.GetBuiltinExtraResource<Material>("Default-Material.mat");
            var m = new Material(baseMat) { name = name, color = c };
            return m;
        }

        static void ApplySurfaces(TerrainData td, string dir, string assetDir, bool overlay, float tile, float size)
        {
            var sj = JsonUtility.FromJson<SurfacesJson>(File.ReadAllText(Path.Combine(dir, "surfaces", "surfaces.json")));
            var ordered = sj.layers.OrderBy(l => l.index).ToArray();
            var layers = new List<TerrainLayer>();
            foreach (var l in ordered)
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

            td.alphamapResolution = Mathf.ClosestPowerOfTwo(Mathf.Clamp(sj.resolution, 16, 4096));
            int res = td.alphamapResolution;
            var maps = new float[res, res, layers.Count];
            if (aerial != null)
            {
                int ai = layers.Count - 1;
                for (int y = 0; y < res; y++) for (int x = 0; x < res; x++) maps[y, x, ai] = 1f;
            }
            else
            {
                var cache = new Dictionary<string, (Color32[] px, int w, int h)>();
                for (int li = 0; li < ordered.Length; li++)
                {
                    var l = ordered[li];
                    if (!cache.TryGetValue(l.control_map, out var img))
                    {
                        var px = LoadPng(Path.Combine(dir, "surfaces", l.control_map), out int w, out int h);
                        img = (px, w, h);
                        cache[l.control_map] = img;
                    }
                    int ch = "RGBA".IndexOf(l.channel);
                    for (int y = 0; y < res; y++)
                    {
                        int sy = y * img.h / res;
                        for (int x = 0; x < res; x++)
                        {
                            var c = img.px[sy * img.w + x * img.w / res];
                            maps[y, x, li] = (ch == 0 ? c.r : ch == 1 ? c.g : ch == 2 ? c.b : c.a) / 255f;
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

        // ---------------------------------------------------------------- trees
        static void ImportTrees(string csvPath, Terrain terrain, string assetDir, GameObject deciduous, GameObject conifer)
        {
            if (!File.Exists(csvPath)) { Debug.LogWarning("[GSPro Builder] trees.csv not found"); return; }
            deciduous = deciduous ? deciduous : PlaceholderTree(assetDir, "Deciduous", false);
            conifer = conifer ? conifer : PlaceholderTree(assetDir, "Conifer", true);
            var protos = new[] { new TreePrototype { prefab = deciduous }, new TreePrototype { prefab = conifer } };
            var refSize = new[] { PrefabSize(deciduous), PrefabSize(conifer) };
            var td = terrain.terrainData;
            td.treePrototypes = protos;
            var size = td.size;
            var list = new List<TreeInstance>();
            var inv = CultureInfo.InvariantCulture;
            foreach (var line in File.ReadLines(csvPath).Skip(1))
            {
                var f = line.Split(',');
                if (f.Length < 6) continue;
                float x = float.Parse(f[0], inv), z = float.Parse(f[2], inv);
                float ht = float.Parse(f[3], inv), crown = float.Parse(f[4], inv);
                int p = f[5].Trim() == "conifer" ? 1 : 0;
                var nx = (x + size.x / 2f) / size.x;
                var nz = (z + size.z / 2f) / size.z;
                if (nx < 0 || nx > 1 || nz < 0 || nz > 1) continue;
                list.Add(new TreeInstance
                {
                    position = new Vector3(nx, 0f, nz),
                    prototypeIndex = p,
                    heightScale = ht / Mathf.Max(refSize[p].y, 0.01f),
                    widthScale = crown * 2f / Mathf.Max(refSize[p].x, 0.01f),
                    rotation = Random.Range(0f, Mathf.PI * 2f),
                    color = Color.white,
                    lightmapColor = Color.white,
                });
            }
            td.SetTreeInstances(list.ToArray(), true);
            terrain.Flush();
            Debug.Log($"[GSPro Builder] Planted {list.Count} trees");
        }

        static Vector2 PrefabSize(GameObject prefab)
        {
            var rs = prefab.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return Vector2.one;
            var b = rs[0].bounds;
            foreach (var r in rs) b.Encapsulate(r.bounds);
            return new Vector2(Mathf.Max(b.size.x, b.size.z), b.size.y);
        }

        /// Single-mesh placeholder tree (terrain trees need the renderer on the prefab root).
        static GameObject PlaceholderTree(string assetDir, string name, bool cone)
        {
            var prefabPath = $"{assetDir}/Tree_{name}.prefab";
            var existing = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (existing) return existing;

            var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            var crownGo = GameObject.CreatePrimitive(cone ? PrimitiveType.Capsule : PrimitiveType.Sphere);
            var combine = new[]
            {
                new CombineInstance { mesh = trunk.GetComponent<MeshFilter>().sharedMesh,
                    transform = Matrix4x4.TRS(new Vector3(0, 2f, 0), Quaternion.identity, new Vector3(0.4f, 2f, 0.4f)) },
                new CombineInstance { mesh = crownGo.GetComponent<MeshFilter>().sharedMesh,
                    transform = cone
                        ? Matrix4x4.TRS(new Vector3(0, 6f, 0), Quaternion.identity, new Vector3(3f, 4.5f, 3f))
                        : Matrix4x4.TRS(new Vector3(0, 6.5f, 0), Quaternion.identity, new Vector3(6f, 5f, 6f)) },
            };
            Object.DestroyImmediate(trunk);
            Object.DestroyImmediate(crownGo);
            var mesh = new Mesh { name = $"Tree_{name}" };
            mesh.CombineMeshes(combine, false, true);
            AssetDatabase.CreateAsset(mesh, $"{assetDir}/Tree_{name}_Mesh.asset");
            var bark = NewLitMaterial(new Color(0.35f, 0.25f, 0.15f), "Bark");
            var leaves = NewLitMaterial(cone ? new Color(0.1f, 0.3f, 0.15f) : new Color(0.2f, 0.45f, 0.15f), $"Leaves_{name}");
            AssetDatabase.CreateAsset(bark, $"{assetDir}/Tree_{name}_Bark.mat");
            AssetDatabase.CreateAsset(leaves, $"{assetDir}/Tree_{name}_Leaves.mat");
            var go = new GameObject($"Tree_{name}");
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterials = new[] { bark, leaves };
            var prefab = PrefabUtility.SaveAsPrefabAsset(go, prefabPath);
            Object.DestroyImmediate(go);
            return prefab;
        }

        // ---------------------------------------------------------------- water
        static void ImportWater(string path, GameObject root, string assetDir, Material mat)
        {
            if (!File.Exists(path)) return;
            var wj = JsonUtility.FromJson<WaterJson>(File.ReadAllText(path));
            if (wj.bodies == null || wj.bodies.Length == 0) return;
            if (mat == null)
            {
                var mp = $"{assetDir}/Water_Placeholder.mat";
                mat = AssetDatabase.LoadAssetAtPath<Material>(mp);
                if (mat == null)
                {
                    mat = NewLitMaterial(new Color(0.08f, 0.22f, 0.35f), "Water_Placeholder");
                    AssetDatabase.CreateAsset(mat, mp);
                }
            }
            var parent = new GameObject("Water");
            parent.transform.SetParent(root.transform, false);
            foreach (var b in wj.bodies)
            {
                var mesh = new Mesh { name = b.name, indexFormat = IndexFormat.UInt32 };
                mesh.vertices = b.vertices.Select(v => v.ToVector3()).ToArray();
                mesh.triangles = b.triangles;
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();
                AssetDatabase.CreateAsset(mesh, $"{assetDir}/{b.name}.asset");
                var go = new GameObject(b.name);
                go.transform.SetParent(parent.transform, false);
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                go.AddComponent<MeshRenderer>().sharedMaterial = mat;
            }
        }

        // ---------------------------------------------------------------- surface meshes
        static void ImportSurfaceMeshes(string objPath, GameObject root, string assetDir)
        {
            if (!File.Exists(objPath)) return;
            var dst = $"{assetDir}/course_surfaces.obj";
            File.Copy(objPath, dst, true);
            AssetDatabase.ImportAsset(dst, ImportAssetOptions.ForceUpdate);
            var model = AssetDatabase.LoadAssetAtPath<GameObject>(dst);
            if (!model) return;
            var inst = (GameObject)PrefabUtility.InstantiatePrefab(model);
            inst.name = "CourseSurfaces";
            inst.transform.SetParent(root.transform, false);
            // OBJ is right-handed with z = south; Unity mirrors X on import, so a 180 deg
            // turn about Y restores x = east, z = north.
            inst.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);
        }

        // ---------------------------------------------------------------- holes
        static void ImportHoles(string path, Terrain terrain, GameObject root)
        {
            if (!File.Exists(path)) { Debug.LogWarning("[GSPro Builder] holes.json not found"); return; }
            var hj = JsonUtility.FromJson<HolesJson>(File.ReadAllText(path));
            var holesRoot = new GameObject("Holes");
            holesRoot.transform.SetParent(root.transform, false);
            foreach (var h in hj.holes)
            {
                var hole = new GameObject($"Hole{h.number:00}");
                hole.transform.SetParent(holesRoot.transform, false);
                var route = hole.AddComponent<HoleRoute>();
                route.number = h.number; route.par = h.par; route.handicap = h.handicap;
                foreach (var t in h.tees)
                    Marker(hole, $"Tee_{t.name}", MarkerKind.Tee, h.number, t.position, terrain).teeName = t.name;
                Marker(hole, "Pin", MarkerKind.Pin, h.number, h.pin, terrain);
                if (h.pin_positions != null)
                    foreach (var pp in h.pin_positions)
                        Marker(hole, $"PinPos_{pp.name}", MarkerKind.PinPosition, h.number, pp.position, terrain).teeName = pp.name;
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
