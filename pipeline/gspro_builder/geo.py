"""Coordinate helpers: WGS84 <-> UTM and the course's local metric frame.

Pure Python (no pyproj) so the pipeline installs with numpy/Pillow/requests only.
UTM formulas follow the standard Krueger/Snyder series (sub-millimetre accuracy
inside a zone), which is far more precise than any LiDAR DEM we feed it.

Frames used by the pipeline:
  * WGS84 lon/lat (degrees)            - GeoJSON, OSM, user input
  * UTM easting/northing (metres)      - data requests and rasters
  * Local course frame (metres)        - origin = centre of the terrain square,
                                         x = east, z = north, y = up (Unity convention)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_A = 6378137.0                 # WGS84 semi-major axis
_F = 1 / 298.257223563         # WGS84 flattening
_K0 = 0.9996
_E2 = _F * (2 - _F)            # eccentricity squared
_EP2 = _E2 / (1 - _E2)


def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) + 1


def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + utm_zone(lon)


def _central_meridian(zone: int) -> float:
    return math.radians((zone - 1) * 6 - 180 + 3)


def lonlat_to_utm(lon: float, lat: float, zone: int | None = None) -> tuple[float, float, int]:
    """Return (easting, northing, zone)."""
    zone = zone or utm_zone(lon)
    phi = math.radians(lat)
    lam = math.radians(lon) - _central_meridian(zone)
    n = _A / math.sqrt(1 - _E2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = _EP2 * math.cos(phi) ** 2
    a = math.cos(phi) * lam
    e4, e6 = _E2 ** 2, _E2 ** 3
    m = _A * ((1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
              - (3 * _E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
              + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
              - (35 * e6 / 3072) * math.sin(6 * phi))
    x = _K0 * n * (a + (1 - t + c) * a ** 3 / 6
                   + (5 - 18 * t + t * t + 72 * c - 58 * _EP2) * a ** 5 / 120) + 500000.0
    y = _K0 * (m + n * math.tan(phi) * (a * a / 2 + (5 - t + 9 * c + 4 * c * c) * a ** 4 / 24
                                        + (61 - 58 * t + t * t + 600 * c - 330 * _EP2) * a ** 6 / 720))
    if lat < 0:
        y += 10000000.0
    return x, y, zone


def utm_to_lonlat(x: float, y: float, zone: int, south: bool = False) -> tuple[float, float]:
    x -= 500000.0
    if south:
        y -= 10000000.0
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    n1 = _A / math.sqrt(1 - _E2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = _EP2 * math.cos(phi1) ** 2
    r1 = _A * (1 - _E2) / (1 - _E2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d * d / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * _EP2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * _EP2 - 3 * c1 * c1) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * _EP2 + 24 * t1 * t1) * d ** 5 / 120) / math.cos(phi1)
    return math.degrees(lon + _central_meridian(zone)), math.degrees(lat)


@dataclass(frozen=True)
class CourseFrame:
    """A square terrain tile centred on (center_lon, center_lat), size_m on a side,
    axis-aligned to the UTM grid of the centre's zone."""
    center_lon: float
    center_lat: float
    size_m: float

    @property
    def zone(self) -> int:
        return utm_zone(self.center_lon)

    @property
    def south(self) -> bool:
        return self.center_lat < 0

    @property
    def epsg(self) -> int:
        return utm_epsg(self.center_lon, self.center_lat)

    @property
    def center_utm(self) -> tuple[float, float]:
        x, y, _ = lonlat_to_utm(self.center_lon, self.center_lat)
        return x, y

    @property
    def utm_bbox(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in UTM metres."""
        cx, cy = self.center_utm
        h = self.size_m / 2
        return cx - h, cy - h, cx + h, cy + h

    def to_local(self, lon: float, lat: float) -> tuple[float, float]:
        """lon/lat -> local (x east, z north) metres from the tile centre."""
        x, y, _ = lonlat_to_utm(lon, lat, self.zone)
        cx, cy = self.center_utm
        return x - cx, y - cy

    def to_lonlat(self, x: float, z: float) -> tuple[float, float]:
        cx, cy = self.center_utm
        return utm_to_lonlat(cx + x, cy + z, self.zone, self.south)

    def local_to_pixel(self, x: float, z: float, res_px: int) -> tuple[float, float]:
        """Local metres -> raster pixel (col, row), row 0 = north edge."""
        h = self.size_m / 2
        return (x + h) / self.size_m * res_px, (h - z) / self.size_m * res_px
