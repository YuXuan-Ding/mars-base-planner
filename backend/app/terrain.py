"""Terrain layers on a working grid, loaded once at startup.

The 200 m DEM is averaged into blocks (default 2x2, about 400 m) so routing stays fast.
Layers per working cell: elevation, gradient, slope, roughness, crater rim distance,
and a modeled rock abundance (placeholder until TES / HiRISE rock data is loaded).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
MARS_R = 3396190.0
DEG_M = 2 * math.pi * MARS_R / 360


def rock_cover_fraction(k: np.ndarray | float, d_m: float) -> np.ndarray | float:
    """Fraction of ground covered by rocks of diameter >= d_m (Golombek & Rapp 1997).

    F_k(D) = k * exp(-q(k) * D), q(k) = 1.79 + 0.152 / k, where k is total rock abundance.
    """
    k = np.maximum(k, 1e-4)
    return k * np.exp(-(1.79 + 0.152 / k) * d_m)


@dataclass
class Grid:
    """A rectangular window of the working grid. Row 0 is the north edge."""
    r0: int
    c0: int
    H: np.ndarray          # elevation, m
    gx: np.ndarray         # dh/d(east), m/m
    gy: np.ndarray         # dh/d(north), m/m
    slope: np.ndarray      # degrees
    rough: np.ndarray      # std of native slope inside the cell, degrees
    rim_dist: np.ndarray   # distance to nearest crater rim, m
    rock_k: np.ndarray     # modeled total rock abundance (0..1)
    dx: np.ndarray         # east-west cell size per row, m
    dy: float              # north-south cell size, m

    @property
    def shape(self) -> tuple[int, int]:
        return self.H.shape


class Terrain:
    def __init__(self, factor: int = 2):
        meta = json.loads((ROOT / "data/jezero_dem_200m.json").read_text())
        rows, cols, res = meta["rows"], meta["cols"], meta["deg_per_px"]
        dem = np.fromfile(ROOT / "data/jezero_dem_200m.i16", dtype="<i2").reshape(rows, cols).astype(np.float32)
        self.meta = meta
        self.native = dem
        self.native_res = res
        self.factor = factor

        # native slope, for roughness
        lat_n = meta["lat_north"] - (np.arange(rows) + 0.5) * res
        dxn = res * DEG_M * np.cos(np.radians(lat_n))
        dyn = res * DEG_M
        gxn = np.gradient(dem, axis=1) / dxn[:, None]
        gyn = -np.gradient(dem, axis=0) / dyn
        sn = np.degrees(np.arctan(np.hypot(gxn, gyn)))

        f = factor
        R, C = rows // f, cols // f
        blk = lambda a: a[: R * f, : C * f].reshape(R, f, C, f)
        H = blk(dem).mean(axis=(1, 3))
        rough = blk(sn).std(axis=(1, 3)) if f > 1 else ndimage.generic_filter(sn, np.std, size=3)

        self.res = res * f
        self.lat = meta["lat_north"] - (np.arange(R) + 0.5) * self.res
        self.lon = meta["lon_west"] + (np.arange(C) + 0.5) * self.res
        self.dx = self.res * DEG_M * np.cos(np.radians(self.lat))
        self.dy = self.res * DEG_M
        gx = np.gradient(H, axis=1) / self.dx[:, None]
        gy = -np.gradient(H, axis=0) / self.dy
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))

        rim_dist, ejecta = self._craters(R, C)
        rough_p95 = float(np.percentile(rough, 95))
        # Placeholder rock abundance: background + crater ejecta + rough ground.
        rock_k = np.clip(0.04 + ejecta + 0.08 * np.clip(rough / rough_p95, 0, 1), 0.02, 0.35)

        self.H, self.gx, self.gy, self.slope = H, gx, gy, slope
        self.rough, self.rough_p95 = rough, rough_p95
        self.rim_dist, self.rock_k = rim_dist, rock_k
        self.shape = (R, C)

    def _craters(self, R: int, C: int) -> tuple[np.ndarray, np.ndarray]:
        feats = json.loads((ROOT / "data/raw/robbins_jezero.json").read_text())["features"]
        rim = np.full((R, C), np.inf, dtype=np.float32)
        ejecta = np.zeros((R, C), dtype=np.float32)
        for ft in feats:
            p = ft["properties"]
            lon, lat, rad = p["lon_e"], p["lat"], p["diamkm"] * 500.0
            reach = 2.5 * rad + 5000.0
            dlat = reach / DEG_M
            dlon = reach / (DEG_M * math.cos(math.radians(lat)))
            r0, r1 = self._row(lat + dlat), self._row(lat - dlat) + 1
            c0, c1 = self._col(lon - dlon), self._col(lon + dlon) + 1
            r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, R), min(c1, C)
            if r0 >= r1 or c0 >= c1:
                continue
            ym = (self.lat[r0:r1, None] - lat) * DEG_M
            xm = (self.lon[None, c0:c1] - lon) * DEG_M * np.cos(np.radians(self.lat[r0:r1, None]))
            r = np.hypot(xm, ym)
            np.minimum(rim[r0:r1, c0:c1], np.abs(r - rad), out=rim[r0:r1, c0:c1])
            # Continuous ejecta is rockiest at the rim and fades within about a radius.
            # Large craters here are old and degraded, so they count for less.
            w = 0.10 if p["diamkm"] < 20 else 0.04
            e = np.where(r > 0.8 * rad, w * np.exp(-np.maximum(r - rad, 0) / (0.4 * rad)), 0)
            np.maximum(ejecta[r0:r1, c0:c1], e, out=ejecta[r0:r1, c0:c1])
        return rim, ejecta

    # coordinates
    def _row(self, lat: float) -> int:
        return int(math.floor((self.meta["lat_north"] - lat) / self.res))

    def _col(self, lon: float) -> int:
        return int(math.floor((lon - self.meta["lon_west"]) / self.res))

    def contains(self, lon: float, lat: float) -> bool:
        m = self.meta
        return m["lon_west"] <= lon <= m["lon_east"] and m["lat_south"] <= lat <= m["lat_north"]

    def rc(self, lon: float, lat: float) -> tuple[int, int]:
        R, C = self.shape
        return min(max(self._row(lat), 0), R - 1), min(max(self._col(lon), 0), C - 1)

    def lonlat(self, r: np.ndarray | int, c: np.ndarray | int):
        return self.lon[c], self.lat[r]

    def grid(self, r0: int = 0, r1: int | None = None, c0: int = 0, c1: int | None = None) -> Grid:
        R, C = self.shape
        r1 = R if r1 is None else r1
        c1 = C if c1 is None else c1
        s = (slice(r0, r1), slice(c0, c1))
        return Grid(r0, c0, self.H[s], self.gx[s], self.gy[s], self.slope[s], self.rough[s],
                    self.rim_dist[s], self.rock_k[s], self.dx[r0:r1], self.dy)

    def surface(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Bilinear elevation from the native 200 m DEM."""
        m = self.meta
        rr = (m["lat_north"] - lats) / self.native_res - 0.5
        cc = (lons - m["lon_west"]) / self.native_res - 0.5
        return ndimage.map_coordinates(self.native, [rr, cc], order=1, mode="nearest")

    def distance_m(self, lon0: float, lat0: float, lon1: float, lat1: float) -> float:
        latm = math.radians((lat0 + lat1) / 2)
        return math.hypot((lon1 - lon0) * DEG_M * math.cos(latm), (lat1 - lat0) * DEG_M)

    def sample(self, lon: float, lat: float) -> dict:
        r, c = self.rc(lon, lat)
        return {
            "elevation_m": round(float(self.surface(np.array([lon]), np.array([lat]))[0]), 1),
            "slope_deg": round(float(self.slope[r, c]), 2),
            "roughness_deg": round(float(self.rough[r, c]), 2),
            "crater_rim_distance_m": round(float(self.rim_dist[r, c])),
            "rock_abundance": round(float(self.rock_k[r, c]), 3),
        }

    def info(self) -> dict:
        m = self.meta
        R, C = self.shape
        return {
            "bounds": {k: m[k] for k in ("lon_west", "lon_east", "lat_south", "lat_north")},
            "source": m["source"],
            "native_resolution_m": round(self.native_res * DEG_M),
            "working_grid": {"rows": R, "cols": C, "resolution_m": round(self.dy)},
            "layers": {
                "elevation": "real (USGS HRSC+MOLA 200 m)",
                "slope, roughness": "real, derived from elevation",
                "crater rims": "real (Robbins database, >= 1 km)",
                "rock abundance": "placeholder: modeled from crater ejecta and roughness",
            },
        }


@lru_cache(maxsize=1)
def get_terrain() -> Terrain:
    return Terrain(factor=int(os.environ.get("MARS_GRID_FACTOR", "2")))
