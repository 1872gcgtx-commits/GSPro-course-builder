using System;

// JsonUtility-compatible mirrors of the files gspro_builder writes.
// Field names match the JSON keys exactly.
namespace GSProCourseBuilder.Editor
{
    [Serializable] public class TerrainJson
    {
        public int resolution;
        public float size_m, min_elev_m, max_elev_m, height_range_m, center_lon, center_lat;
        public int utm_epsg;
        public string raw_file, byte_order, row_order;
        public bool unity_flip_vertically;
    }

    [Serializable] public class SurfaceLayerJson { public int index; public string name; public string control_map; public string channel; }
    [Serializable] public class SurfacesJson { public int resolution; public float size_m; public string row_order; public int control_maps; public SurfaceLayerJson[] layers; }

    [Serializable] public class WaterBodyJson { public string name; public int hole; public float level_y; public V3[] vertices; public int[] triangles; }
    [Serializable] public class WaterJson { public float base_elev_m; public WaterBodyJson[] bodies; }

    [Serializable] public class PinPositionJson { public string name; public V3 position; public float slope_pct; }

    [Serializable] public class V3
    {
        public float x, y, z;
        public V3() { }
        public V3(UnityEngine.Vector3 v) { x = v.x; y = v.y; z = v.z; }
        public UnityEngine.Vector3 ToVector3() => new UnityEngine.Vector3(x, y, z);
    }

    [Serializable] public class TeeJson { public string name; public V3 position; public int yards; }

    [Serializable] public class HoleJson
    {
        public int number;
        public string name;
        public int par;
        public bool par_estimated;
        public int handicap;
        public TeeJson[] tees;
        public V3 pin;
        public PinPositionJson[] pin_positions;
        public V3[] aim_points;
        public V3 green_center;
        public float elevation_change_m;
    }

    [Serializable] public class HolesJson
    {
        public string course;
        public string frame;
        public int par_total;
        public HoleJson[] holes;
    }
}
