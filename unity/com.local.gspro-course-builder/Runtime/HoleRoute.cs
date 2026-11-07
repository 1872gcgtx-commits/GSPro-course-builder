using System.Collections.Generic;
using UnityEngine;

namespace GSProCourseBuilder
{
    /// <summary>Draws the hole's routing line (tee -> aim points -> pin) and yardage.</summary>
    [ExecuteAlways]
    public class HoleRoute : MonoBehaviour
    {
        public int number;
        public int par;
        public int handicap;

        void OnDrawGizmos()
        {
            var markers = GetComponentsInChildren<HoleMarker>();
            HoleMarker tee = null, pin = null;
            var aims = new List<HoleMarker>();
            foreach (var m in markers)
            {
                if (m.kind == MarkerKind.Tee && tee == null) tee = m;
                else if (m.kind == MarkerKind.Pin) pin = m;
                else if (m.kind == MarkerKind.AimPoint) aims.Add(m);
            }
            if (tee == null || pin == null) return;
            aims.Sort((a, b) => a.order.CompareTo(b.order));
            Gizmos.color = Color.white;
            var prev = tee.transform.position + Vector3.up;
            foreach (var a in aims)
            {
                var p = a.transform.position + Vector3.up;
                Gizmos.DrawLine(prev, p);
                prev = p;
            }
            Gizmos.DrawLine(prev, pin.transform.position + Vector3.up);
        }

        /// <summary>Playing length from the first tee, via aim points, to the pin, in yards (horizontal).</summary>
        public float YardsFrom(HoleMarker tee)
        {
            var markers = GetComponentsInChildren<HoleMarker>();
            var aims = new List<HoleMarker>();
            HoleMarker pin = null;
            foreach (var m in markers)
            {
                if (m.kind == MarkerKind.AimPoint) aims.Add(m);
                else if (m.kind == MarkerKind.Pin) pin = m;
            }
            if (pin == null) return 0f;
            aims.Sort((a, b) => a.order.CompareTo(b.order));
            float d = 0f;
            var prev = Flat(tee.transform.position);
            foreach (var a in aims) { var p = Flat(a.transform.position); d += Vector2.Distance(prev, p); prev = p; }
            d += Vector2.Distance(prev, Flat(pin.transform.position));
            return d * 1.0936133f;
        }

        static Vector2 Flat(Vector3 v) => new Vector2(v.x, v.z);
    }
}
