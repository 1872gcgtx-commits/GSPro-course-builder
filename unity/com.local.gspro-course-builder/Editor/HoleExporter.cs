using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace GSProCourseBuilder.Editor
{
    /// <summary>Writes the (possibly hand-edited) hole markers in the open scene back
    /// to a holes.json-format file, with yardages recomputed. Use it as the reference
    /// sheet when entering tees/pins/aim points in GreenKeeper.</summary>
    public static class HoleExporter
    {
        [MenuItem("GSPro Builder/Export Hole Markers...")]
        static void ExportMenu()
        {
            var path = EditorUtility.SaveFilePanel("Export holes", "", "holes_edited.json", "json");
            if (!string.IsNullOrEmpty(path)) Export(path);
        }

        public static void Export(string path)
        {
            var routes = Object.FindObjectsByType<HoleRoute>(FindObjectsSortMode.None).OrderBy(r => r.number).ToList();
            if (routes.Count == 0) { EditorUtility.DisplayDialog("GSPro Builder", "No HoleRoute objects in the scene.", "OK"); return; }
            var holes = new List<HoleJson>();
            foreach (var r in routes)
            {
                var markers = r.GetComponentsInChildren<HoleMarker>();
                var tees = markers.Where(m => m.kind == MarkerKind.Tee).ToList();
                var pin = markers.FirstOrDefault(m => m.kind == MarkerKind.Pin);
                var green = markers.FirstOrDefault(m => m.kind == MarkerKind.GreenCenter);
                if (pin == null || tees.Count == 0) { Debug.LogWarning($"[GSPro Builder] Hole {r.number}: needs a tee and a pin"); continue; }
                holes.Add(new HoleJson
                {
                    number = r.number,
                    name = r.gameObject.name,
                    par = r.par,
                    handicap = r.handicap,
                    tees = tees.Select(t => new TeeJson
                    {
                        name = string.IsNullOrEmpty(t.teeName) ? t.name : t.teeName,
                        position = new V3(t.transform.position),
                        yards = Mathf.RoundToInt(r.YardsFrom(t)),
                    }).ToArray(),
                    pin = new V3(pin.transform.position),
                    pin_positions = markers.Where(m => m.kind == MarkerKind.PinPosition)
                                           .Select(m => new PinPositionJson { name = m.teeName, position = new V3(m.transform.position) })
                                           .ToArray(),
                    aim_points = markers.Where(m => m.kind == MarkerKind.AimPoint).OrderBy(m => m.order)
                                        .Select(m => new V3(m.transform.position)).ToArray(),
                    green_center = new V3((green ? green : pin).transform.position),
                    elevation_change_m = pin.transform.position.y - tees[0].transform.position.y,
                });
            }
            var doc = new HolesJson
            {
                course = SceneManagerName(),
                frame = "Unity metres: x east, y up from terrain base, z north; origin = terrain centre",
                par_total = holes.Sum(h => h.par),
                holes = holes.ToArray(),
            };
            File.WriteAllText(path, JsonUtility.ToJson(doc, true));
            Debug.Log($"[GSPro Builder] Wrote {holes.Count} holes to {path}");
        }

        static string SceneManagerName() => UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
    }
}
