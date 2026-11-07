using UnityEngine;

namespace GSProCourseBuilder
{
    public enum MarkerKind { Tee, Pin, AimPoint, GreenCenter, PinPosition }

    /// <summary>A tee / pin / aim point for one hole. Move these in the Scene view,
    /// then use GSPro Builder > Export Hole Markers to write holes_edited.json.</summary>
    [ExecuteAlways]
    public class HoleMarker : MonoBehaviour
    {
        public int hole;
        public MarkerKind kind;
        public string teeName;     // Tee: tee-set name; PinPosition: front / middle / back
        public int order;          // AimPoint order along the hole

        static readonly Color TeeColor = new Color(0.2f, 0.5f, 1f);
        static readonly Color PinColor = new Color(1f, 0.15f, 0.15f);
        static readonly Color AimColor = new Color(1f, 0.85f, 0.1f);

        void OnDrawGizmos()
        {
            var p = transform.position;
            switch (kind)
            {
                case MarkerKind.Tee:
                    Gizmos.color = TeeColor;
                    Gizmos.DrawCube(p + Vector3.up * 0.5f, new Vector3(2f, 1f, 2f));
                    break;
                case MarkerKind.Pin:
                    Gizmos.color = PinColor;
                    Gizmos.DrawLine(p, p + Vector3.up * 2.4f);
                    Gizmos.DrawCube(p + new Vector3(0.3f, 2.2f, 0f), new Vector3(0.6f, 0.4f, 0.05f));
                    break;
                case MarkerKind.PinPosition:
                    Gizmos.color = new Color(1f, 0.5f, 0.5f);
                    Gizmos.DrawWireSphere(p, 0.6f);
                    break;
                case MarkerKind.AimPoint:
                    Gizmos.color = AimColor;
                    Gizmos.DrawWireSphere(p + Vector3.up, 3f);
                    break;
                default:
                    Gizmos.color = Color.white;
                    Gizmos.DrawWireSphere(p, 1f);
                    break;
            }
        }
    }
}
