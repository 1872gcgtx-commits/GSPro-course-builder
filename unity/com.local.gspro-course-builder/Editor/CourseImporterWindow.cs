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
    ///   trees.csv (+ shrubs.csv)                -> terrain trees by species (your prefabs, or placeholders)
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
        bool useOverlay, importHoles = true, importTrees = true, importWater = true, importMeshes, importHomes = true;
        float layerTileSize = 4f;
        GameObject deciduousPrefab, coniferPrefab;
        static readonly string[] Species = { "live_oak", "ashe_juniper", "cedar_elm", "pecan", "bald_cypress",
                                             "mesquite", "hackberry", "brush", "juniper_shrub" };
        GameObject[] speciesPrefabs = new GameObject[Species.Length];
        bool showSpecies;
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
                showSpecies = EditorGUILayout.Foldout(showSpecies, "  Prefab per species (optional)");
                if (showSpecies)
                    for (int i = 0; i < Species.Length; i++)
                        speciesPrefabs[i] = (GameObject)EditorGUILayout.ObjectField("    " + Species[i], speciesPrefabs[i], typeof(GameObject), false);
            }
            importMeshes = EditorGUILayout.Toggle("Course surface meshes (OBJ)", importMeshes);
            importHomes = EditorGUILayout.Toggle("Homes (homes/homes.obj)", importHomes);
            EditorGUILayout.HelpBox("Empty slots get placeholder assets (flat-colour layers, simple trees, " +
                                    "blue water). Swap in real grass/sand textures, SpeedTree/tree prefabs " +
                                    "and your water shader afterwards - re-importing keeps your layer textures.",
                                    MessageType.Info);
            GUI.enabled = Directory.Exists(outputDir);
            if (GUILayout.Button("Import", GUILayout.Height(30)))
                Import(new Options
                {
                    dir = outputDir, overlay = useOverlay, holes = importHoles, trees = importTrees,
                    water = importWater, meshes = importMeshes, homes = importHomes, tile = layerTileSize,
                    deciduous = deciduousPrefab, conifer = coniferPrefab, waterMat = waterMaterial,
                    species = (GameObject[])speciesPrefabs.Clone(),
                });
            GUI.enabled = true;
        }

        public class Options
        {
            public string dir;
            public bool overlay, holes = true, trees = true, water = true, meshes, homes = true;
            public float tile = 4f;
            public GameObject deciduous, conifer;
            public GameObject[] species;
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
                    ImportTrees(o.dir, terrain, assetDir, o);
                }
                if (o.homes)
                {
                    EditorUtility.DisplayProgressBar("GSPro Builder", "Homes", 0.6f);
                    ImportHomes(o.dir, root, assetDir);
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
        // Placeholder shape per species: crown (width, height, centre height) for a 10 m tree, crown shape, colour
        static readonly Dictionary<string, (Vector3 crown, bool cone, Color leaf)> SpeciesLook = new Dictionary<string, (Vector3, bool, Color)>
        {
            { "live_oak",      (new Vector3(12f, 6f, 6.0f), false, new Color(0.16f, 0.30f, 0.12f)) },
            { "ashe_juniper",  (new Vector3(4.5f, 8.5f, 5.2f), true, new Color(0.11f, 0.22f, 0.15f)) },
            { "cedar_elm",     (new Vector3(7f, 7f, 6.3f), false, new Color(0.26f, 0.40f, 0.16f)) },
            { "pecan",         (new Vector3(10f, 7.5f, 6.2f), false, new Color(0.30f, 0.44f, 0.18f)) },
            { "bald_cypress",  (new Vector3(4.5f, 8.5f, 5.6f), true, new Color(0.24f, 0.36f, 0.18f)) },
            { "mesquite",      (new Vector3(9f, 5f, 6.5f), false, new Color(0.36f, 0.44f, 0.24f)) },
            { "hackberry",     (new Vector3(8f, 7f, 6.0f), false, new Color(0.28f, 0.40f, 0.18f)) },
            { "brush",         (new Vector3(9f, 7f, 4.5f), false, new Color(0.30f, 0.40f, 0.20f)) },
            { "juniper_shrub", (new Vector3(7f, 9f, 5.0f), true, new Color(0.12f, 0.24f, 0.16f)) },
            { "deciduous",     (new Vector3(6f, 5f, 6.5f), false, new Color(0.2f, 0.45f, 0.15f)) },
            { "conifer",       (new Vector3(3f, 4.5f, 6.0f), true, new Color(0.1f, 0.3f, 0.15f)) },
        };

        static void ImportTrees(string dir, Terrain terrain, string assetDir, Options o)
        {
            var csvPath = Path.Combine(dir, "trees.csv");
            if (!File.Exists(csvPath)) { Debug.LogWarning("[GSPro Builder] trees.csv not found"); return; }
            var td = terrain.terrainData;
            var size = td.size;
            var inv = CultureInfo.InvariantCulture;
            var protoIndex = new Dictionary<string, int>();
            var protos = new List<TreePrototype>();
            var refSize = new List<Vector2>();
            int Proto(string key, int variant)
            {
                GameObject prefab = null;
                int si = System.Array.IndexOf(Species, key);
                if (si >= 0 && o.species != null && si < o.species.Length) prefab = o.species[si];
                bool conif = key == "ashe_juniper" || key == "bald_cypress" || key == "juniper_shrub" || key == "conifer";
                if (!prefab) prefab = conif ? o.conifer : o.deciduous;
                if (!prefab)
                {
                    // generated species models (output/tree_models/<species>_<n>.obj), 3 variants each
                    var vkey = $"{key}_{variant}";
                    if (protoIndex.TryGetValue(vkey, out var vi)) return vi;
                    prefab = GeneratedTree(dir, assetDir, key, variant);
                    if (prefab)
                    {
                        protos.Add(new TreePrototype { prefab = prefab });
                        refSize.Add(PrefabSize(prefab));
                        protoIndex[vkey] = protos.Count - 1;
                        return protos.Count - 1;
                    }
                }
                if (protoIndex.TryGetValue(key, out var i)) return i;
                if (!prefab) prefab = PlaceholderTree(assetDir, key);
                protos.Add(new TreePrototype { prefab = prefab });
                refSize.Add(PrefabSize(prefab));
                protoIndex[key] = protos.Count - 1;
                return protos.Count - 1;
            }
            var list = new List<TreeInstance>();
            void Plant(string path, bool shrubs)
            {
                if (!File.Exists(path)) return;
                var lines = File.ReadLines(path).ToList();
                if (lines.Count < 2) return;
                var head = lines[0].Split(',').Select(h => h.Trim()).ToList();
                int C(string name) => head.IndexOf(name);
                int cx = C("x"), cz = C("z"), ch = C("height_m"), cr = C("crown_radius_m"), ct = C("type"),
                    cs = C(shrubs ? "kind" : "species"), crot = C("rotation_deg"),
                    tr = C("tint_r"), tg = C("tint_g"), tb = C("tint_b");
                foreach (var line in lines.Skip(1))
                {
                    var f = line.Split(',');
                    if (f.Length < head.Count) continue;
                    float x = float.Parse(f[cx], inv), z = float.Parse(f[cz], inv), ht = float.Parse(f[ch], inv);
                    float crown = cr >= 0 ? float.Parse(f[cr], inv) : Mathf.Max(0.6f, ht * 0.6f);
                    var key = cs >= 0 && f[cs].Trim().Length > 0 ? f[cs].Trim() : (ct >= 0 ? f[ct].Trim() : "deciduous");
                    var nx = (x + size.x / 2f) / size.x;
                    var nz = (z + size.z / 2f) / size.z;
                    if (nx < 0 || nx > 1 || nz < 0 || nz > 1) continue;
                    int p = Proto(key, Mathf.Abs((int)(x * 7 + z * 13)) % 3);
                    var tint = Color.white;
                    if (tr >= 0 && f[tr].Length > 0)
                        tint = new Color(float.Parse(f[tr], inv), float.Parse(f[tg], inv), float.Parse(f[tb], inv), 1f);
                    float rot = crot >= 0 && f[crot].Length > 0 ? float.Parse(f[crot], inv) * Mathf.Deg2Rad : Random.Range(0f, Mathf.PI * 2f);
                    list.Add(new TreeInstance
                    {
                        position = new Vector3(nx, 0f, nz),
                        prototypeIndex = p,
                        heightScale = ht / Mathf.Max(refSize[p].y, 0.01f),
                        widthScale = crown * 2f / Mathf.Max(refSize[p].x, 0.01f),
                        rotation = rot,
                        color = tint,
                        lightmapColor = Color.white,
                    });
                }
            }
            Plant(csvPath, false);
            Plant(Path.Combine(dir, "shrubs.csv"), true);
            td.treePrototypes = protos.ToArray();
            td.SetTreeInstances(list.ToArray(), true);
            terrain.Flush();
            Debug.Log($"[GSPro Builder] Planted {list.Count} trees and shrubs ({protos.Count} species)");
        }

        static Vector2 PrefabSize(GameObject prefab)
        {
            var rs = prefab.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return Vector2.one;
            var b = rs[0].bounds;
            foreach (var r in rs) b.Encapsulate(r.bounds);
            return new Vector2(Mathf.Max(b.size.x, b.size.z), b.size.y);
        }

        /// Prefab from a generated species model: OBJ copied into the project, child meshes
        /// combined onto the root (terrain trees need the renderer there), bark + cutout leaves.
        static GameObject GeneratedTree(string dir, string assetDir, string species, int variant)
        {
            var src = Path.Combine(dir, "tree_models");
            var obj = Path.Combine(src, $"{species}_{variant}.obj");
            if (!File.Exists(obj)) return null;
            var dst = $"{assetDir}/TreeModels";
            var prefabPath = $"{dst}/Tree_{species}_{variant}.prefab";
            var existing = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (existing) return existing;
            Directory.CreateDirectory(dst);
            foreach (var f in new[] { $"{species}_{variant}.obj", $"{species}.mtl", $"leaves_{species}.png", "bark.png" })
            {
                var from = Path.Combine(src, f);
                if (File.Exists(from) && !File.Exists($"{dst}/{f}")) File.Copy(from, $"{dst}/{f}");
            }
            AssetDatabase.Refresh();
            var leafTexPath = $"{dst}/leaves_{species}.png";
            if (AssetImporter.GetAtPath(leafTexPath) is TextureImporter ti)
            {
                ti.alphaIsTransparency = true; ti.mipMapsPreserveCoverage = true; ti.alphaTestReferenceValue = 0.45f;
                ti.SaveAndReimport();
            }
            var model = AssetDatabase.LoadAssetAtPath<GameObject>($"{dst}/{species}_{variant}.obj");
            if (!model) return null;
            // combine every child mesh into one mesh; submesh 0 = bark, 1 = leaves (by material name)
            var bark = new List<CombineInstance>(); var leaves = new List<CombineInstance>();
            foreach (var mf in model.GetComponentsInChildren<MeshFilter>())
            {
                var mr = mf.GetComponent<MeshRenderer>();
                for (int s = 0; s < mf.sharedMesh.subMeshCount; s++)
                {
                    var mname = mr && s < mr.sharedMaterials.Length && mr.sharedMaterials[s] ? mr.sharedMaterials[s].name : "";
                    var ci = new CombineInstance { mesh = mf.sharedMesh, subMeshIndex = s, transform = mf.transform.localToWorldMatrix };
                    (mname.StartsWith("Leaves") ? leaves : bark).Add(ci);
                }
            }
            Mesh Part(List<CombineInstance> l) { var m = new Mesh { indexFormat = IndexFormat.UInt32 }; m.CombineMeshes(l.ToArray(), true, true); return m; }
            var mesh = new Mesh { name = $"Tree_{species}_{variant}", indexFormat = IndexFormat.UInt32 };
            mesh.CombineMeshes(new[] { new CombineInstance { mesh = Part(bark) }, new CombineInstance { mesh = Part(leaves) } }, false, false);
            mesh.RecalculateBounds();
            AssetDatabase.CreateAsset(mesh, $"{dst}/Tree_{species}_{variant}_Mesh.asset");
            var barkMat = AssetDatabase.LoadAssetAtPath<Material>($"{dst}/Tree_Bark.mat");
            if (!barkMat)
            {
                barkMat = NewLitMaterial(Color.white, "Tree_Bark");
                SetMainTexture(barkMat, AssetDatabase.LoadAssetAtPath<Texture2D>($"{dst}/bark.png"));
                AssetDatabase.CreateAsset(barkMat, $"{dst}/Tree_Bark.mat");
            }
            var leafMat = AssetDatabase.LoadAssetAtPath<Material>($"{dst}/Tree_{species}_Leaves.mat");
            if (!leafMat)
            {
                leafMat = NewLitMaterial(Color.white, $"Tree_{species}_Leaves");
                SetMainTexture(leafMat, AssetDatabase.LoadAssetAtPath<Texture2D>(leafTexPath));
                MakeCutout(leafMat, 0.45f);
                AssetDatabase.CreateAsset(leafMat, $"{dst}/Tree_{species}_Leaves.mat");
            }
            var go = new GameObject($"Tree_{species}_{variant}");
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterials = new[] { barkMat, leafMat };
            var prefab = PrefabUtility.SaveAsPrefabAsset(go, prefabPath);
            Object.DestroyImmediate(go);
            return prefab;
        }

        static void SetMainTexture(Material m, Texture2D t)
        {
            if (!t) return;
            if (m.HasProperty("_BaseColorMap")) m.SetTexture("_BaseColorMap", t);   // HDRP
            if (m.HasProperty("_BaseMap")) m.SetTexture("_BaseMap", t);             // URP
            if (m.HasProperty("_MainTex")) m.SetTexture("_MainTex", t);             // Built-in
        }

        /// Alpha-tested (cutout), double-sided foliage in any render pipeline.
        static void MakeCutout(Material m, float cutoff)
        {
            if (m.HasProperty("_Cutoff")) m.SetFloat("_Cutoff", cutoff);
            if (m.HasProperty("_AlphaClip")) { m.SetFloat("_AlphaClip", 1f); m.EnableKeyword("_ALPHATEST_ON"); }          // URP
            if (m.HasProperty("_AlphaCutoffEnable")) { m.SetFloat("_AlphaCutoffEnable", 1f); m.SetFloat("_AlphaCutoff", cutoff); m.EnableKeyword("_ALPHATEST_ON"); } // HDRP
            if (m.HasProperty("_Mode")) { m.SetFloat("_Mode", 1f); m.EnableKeyword("_ALPHATEST_ON"); }                    // Built-in Standard
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);
            if (m.HasProperty("_CullMode")) m.SetFloat("_CullMode", 0f);
            if (m.HasProperty("_DoubleSidedEnable")) { m.SetFloat("_DoubleSidedEnable", 1f); m.doubleSidedGI = true; }
            m.renderQueue = (int)RenderQueue.AlphaTest;
        }

        /// Single-mesh placeholder tree shaped and coloured for its species
        /// (terrain trees need the renderer on the prefab root). Swap in real models later.
        static GameObject PlaceholderTree(string assetDir, string name)
        {
            var prefabPath = $"{assetDir}/Tree_{name}.prefab";
            var existing = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
            if (existing) return existing;
            if (!SpeciesLook.TryGetValue(name, out var look)) look = SpeciesLook["deciduous"];
            var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            var crownGo = GameObject.CreatePrimitive(look.cone ? PrimitiveType.Capsule : PrimitiveType.Sphere);
            float trunkTop = look.crown.z - look.crown.y / 2f + 0.5f;
            var combine = new[]
            {
                new CombineInstance { mesh = trunk.GetComponent<MeshFilter>().sharedMesh,
                    transform = Matrix4x4.TRS(new Vector3(0, trunkTop / 2f, 0), Quaternion.identity, new Vector3(0.45f, trunkTop / 2f, 0.45f)) },
                new CombineInstance { mesh = crownGo.GetComponent<MeshFilter>().sharedMesh,
                    transform = Matrix4x4.TRS(new Vector3(0, look.crown.z, 0), Quaternion.identity,
                                              new Vector3(look.crown.x, look.cone ? look.crown.y / 2f : look.crown.y, look.crown.x)) },
            };
            Object.DestroyImmediate(trunk);
            Object.DestroyImmediate(crownGo);
            var mesh = new Mesh { name = $"Tree_{name}" };
            mesh.CombineMeshes(combine, false, true);
            AssetDatabase.CreateAsset(mesh, $"{assetDir}/Tree_{name}_Mesh.asset");
            var bark = AssetDatabase.LoadAssetAtPath<Material>($"{assetDir}/Tree_Bark.mat");
            if (!bark) { bark = NewLitMaterial(new Color(0.33f, 0.26f, 0.19f), "Bark"); AssetDatabase.CreateAsset(bark, $"{assetDir}/Tree_Bark.mat"); }
            var leaves = NewLitMaterial(look.leaf, $"Leaves_{name}");
            AssetDatabase.CreateAsset(leaves, $"{assetDir}/Tree_{name}_Leaves.mat");
            var go = new GameObject($"Tree_{name}");
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            go.AddComponent<MeshRenderer>().sharedMaterials = new[] { bark, leaves };
            var prefab = PrefabUtility.SaveAsPrefabAsset(go, prefabPath);
            Object.DestroyImmediate(go);
            return prefab;
        }

        // ---------------------------------------------------------------- homes
        /// homes/homes.obj + .mtl + textures: copied into the project and placed like the
        /// course surface meshes (OBJ is x east, z south; Unity's importer mirrors X, so the
        /// model is turned 180 degrees about Y to restore east/north).
        static void ImportHomes(string dir, GameObject root, string assetDir)
        {
            var src = Path.Combine(dir, "homes");
            if (!File.Exists(Path.Combine(src, "homes.obj"))) return;
            var dst = $"{assetDir}/Homes";
            Directory.CreateDirectory(dst);
            foreach (var f in Directory.GetFiles(src))
            {
                var ext = Path.GetExtension(f).ToLowerInvariant();
                if (ext == ".obj" || ext == ".mtl" || ext == ".png") File.Copy(f, $"{dst}/{Path.GetFileName(f)}", true);
            }
            AssetDatabase.Refresh();
            var model = AssetDatabase.LoadAssetAtPath<GameObject>($"{dst}/homes.obj");
            if (!model) { Debug.LogWarning("[GSPro Builder] homes.obj did not import"); return; }
            var old = root.transform.Find("Homes");
            if (old) Object.DestroyImmediate(old.gameObject);
            var inst = (GameObject)PrefabUtility.InstantiatePrefab(model);
            inst.name = "Homes";
            inst.transform.SetParent(root.transform, false);
            inst.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);
            foreach (var r in inst.GetComponentsInChildren<Renderer>())
                r.gameObject.isStatic = true;
            Debug.Log($"[GSPro Builder] Homes placed ({inst.transform.childCount} tiles)");
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
