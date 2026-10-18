"""Accident odds on every map tile, and the site layout the simulator runs on.

Each 400 m tile of the working grid carries odds for the work that can happen on it:

  wheel_per_km      chance per km driven that a hauler (or the dozer) breaks running gear
                    on a rock. From the cover of rocks >= 0.25 m (hauler clearance), raised
                    on sloped ground.
  hard_per_km       hard-rock bands a TBM meets per km bored under the tile. From rock
                    abundance and nearness to crater rims (impact breccia, ejecta).
  bearing_mult      how much faster TBM main bearings start failing under the tile
                    (shock loads in blocky rock). Same geology index as hard_per_km.
  dozer_per_1000m3  chance per 1000 m3 graded that the dozer hits a buried boulder and
                    damages its tracks. From the cover of rocks >= 0.5 m.

The odds are public: the planner can see the map. What is hidden is how the dice fall,
e.g. where exactly the hard bands are under a tile.

The rock layer is still a placeholder (see app/terrain.py), so these odds are planning
assumptions tied to real terrain, not measurements.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.terrain import DEG_M, Terrain, get_terrain, rock_cover_fraction

K_REF = 0.06          # typical rock abundance on the Jezero floor; geology index 1 here
RIM_SCALE_M = 2500.0  # crater rims raise the hard-rock odds within a few km
TUNNEL_LEN_M = 4500.0 # tunnel lines are laid out this far past each portal

BASIS = {
    "wheel_per_km": "per km driven: wheel_damage_per_km_pct x cover of rocks >= 0.25 m (%) x (1 + slope / wheel_slope_ref_deg)",
    "hard_per_km": "hard-rock bands per km bored: hard_bands_per_km x geology index",
    "bearing_mult": "main-bearing failure onset multiplier: 0.5 + 0.5 x geology index",
    "dozer_per_1000m3": "per 1000 m3 graded: dozer_damage_per_1000m3_pct x cover of rocks >= 0.5 m (%)",
    "geology_index": "(rock abundance / 0.06) x (1 + 2 exp(-crater rim distance / 2.5 km)), clipped to 0.3-4",
    "rock_data": "placeholder rock abundance (crater ejecta + roughness), real slope and crater rims",
}


def odds(rock_k, slope, rim_m, p) -> dict:
    """Odds for arrays (or scalars) of terrain values. `p` is sim Params."""
    rock_k = np.asarray(rock_k, dtype=float)
    rim = np.where(np.isfinite(rim_m), rim_m, 1e9)
    geo = np.clip(rock_k / K_REF * (1 + 2 * np.exp(-rim / RIM_SCALE_M)), 0.3, 4.0)
    c25 = rock_cover_fraction(rock_k, 0.25) * 100
    c50 = rock_cover_fraction(rock_k, 0.5) * 100
    return {
        "wheel_per_km": p.wheel_damage_per_km_pct * c25 * (1 + np.asarray(slope) / p.wheel_slope_ref_deg),
        "hard_per_km": p.hard_bands_per_km * geo,
        "bearing_mult": 0.5 + 0.5 * geo,
        "dozer_per_1000m3": p.dozer_damage_per_1000m3_pct * c50,
        "geology_index": geo,
        "rock_cover_025_pct": c25,
    }


def odds_grid(p, lon0: float, lat0: float, lon1: float, lat1: float, max_cells: int = 40_000) -> dict:
    """Odds layers over a lon/lat window, for drawing on the map."""
    t = get_terrain()
    r0, c0 = t.rc(min(lon0, lon1), max(lat0, lat1))
    r1, c1 = t.rc(max(lon0, lon1), min(lat0, lat1))
    r1, c1 = r1 + 1, c1 + 1
    if (r1 - r0) * (c1 - c0) > max_cells:
        raise ValueError(f"window too large: {(r1 - r0) * (c1 - c0)} tiles (max {max_cells})")
    s = (slice(r0, r1), slice(c0, c1))
    o = odds(t.rock_k[s], t.slope[s], t.rim_dist[s], p)
    half = t.res / 2
    return {
        "bounds": {"lon_west": float(t.lon[c0] - half), "lon_east": float(t.lon[c1 - 1] + half),
                   "lat_north": float(t.lat[r0] + half), "lat_south": float(t.lat[r1 - 1] - half)},
        "rows": r1 - r0, "cols": c1 - c0, "resolution_m": round(t.dy),
        "row_order": "north to south",
        "layers": {k: np.round(v, 6).tolist() for k, v in o.items()},
        "basis": BASIS,
    }


# site layout

@dataclass
class Site:
    tiles: list            # [{r, c, lon, lat, <odds>}]
    tunnels: dict          # T -> {portal, ore, access_m, length_m, dir_m, segs: [(start_m, end_m, tile)]}
    routes: dict           # name -> {km, coords, segs: [(km, tile)], hits_per_trip, rock_cover_pct}
    pad: dict              # {center, tiles}
    plant: list
    yard: list

    def tunnel_tile(self, tunnel: str, m: float) -> dict:
        segs = self.tunnels[tunnel]["segs"]
        i = bisect.bisect_right([a for a, _, _ in segs], m) - 1
        return self.tiles[segs[max(0, min(i, len(segs) - 1))][2]]

    def tunnel_point(self, tunnel: str, m: float) -> list:
        tn = self.tunnels[tunnel]
        lon0, lat0 = tn["portal"]
        ex, ny = tn["dir_m"]
        return _offset(lon0, lat0, ex * m, ny * m)

    def pad_tile(self, frac_done: float) -> dict:
        ts = self.pad["tiles"]
        return self.tiles[ts[min(int(frac_done * len(ts)), len(ts) - 1)]]

    def max_hard_per_km(self, tunnel: str) -> float:
        return max(self.tiles[ti]["hard_per_km"] for _, _, ti in self.tunnels[tunnel]["segs"])


def _offset(lon: float, lat: float, east_m: float, north_m: float) -> list:
    return [lon + east_m / (DEG_M * math.cos(math.radians(lat))), lat + north_m / DEG_M]


def _metres(a, b) -> tuple[float, float]:
    latm = math.radians((a[1] + b[1]) / 2)
    return (b[0] - a[0]) * DEG_M * math.cos(latm), (b[1] - a[1]) * DEG_M


class _Tiles:
    def __init__(self, t: Terrain, p):
        self.t, self.p, self.index, self.list = t, p, {}, []

    def at(self, lon: float, lat: float) -> int:
        r, c = self.t.rc(lon, lat)
        if (r, c) not in self.index:
            t = self.t
            o = odds(t.rock_k[r, c], t.slope[r, c], t.rim_dist[r, c], self.p)
            self.index[(r, c)] = len(self.list)
            self.list.append({"r": int(r), "c": int(c), "lon": round(float(t.lon[c]), 5), "lat": round(float(t.lat[r]), 5),
                              **{k: round(float(v), 6) for k, v in o.items()}})
        return self.index[(r, c)]


def _tunnel(tiles: _Tiles, portal, ore) -> dict:
    ex, ny = _metres(portal, ore)
    access = math.hypot(ex, ny)
    if access < 50:
        raise ValueError("portal and ore target are too close")
    ex, ny = ex / access, ny / access
    segs = []
    step = 20.0
    for i in range(int(TUNNEL_LEN_M / step)):
        m = i * step
        ti = tiles.at(*_offset(*portal, ex * m, ny * m))
        if segs and segs[-1][2] == ti:
            segs[-1][1] = m + step
        else:
            segs.append([m, m + step, ti])
    return {"portal": list(portal), "ore": list(ore), "access_m": round(access, 1), "length_m": TUNNEL_LEN_M,
            "dir_m": (ex, ny), "segs": [tuple(s) for s in segs]}


def _route(tiles: _Tiles, coords: list) -> dict:
    """Split a route polyline into km per tile, in driving order."""
    segs: list[list] = []
    km = 0.0
    for a, b in zip(coords[:-1], coords[1:]):
        ex, ny = _metres(a, b)
        L = math.hypot(ex, ny)
        n = max(1, int(L / 25))
        for j in range(n):
            f = (j + 0.5) / n
            ti = tiles.at(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            d = L / n / 1000
            km += d
            if segs and segs[-1][1] == ti:
                segs[-1][0] += d
            else:
                segs.append([d, ti])
    hits = sum(d * tiles.list[ti]["wheel_per_km"] for d, ti in segs)   # one way
    cover = sum(d * tiles.list[ti]["rock_cover_025_pct"] for d, ti in segs) / max(km, 1e-9)
    return {"km": round(km, 2), "coords": coords, "segs": [(round(d, 4), ti) for d, ti in segs],
            "hits_per_trip": 2 * hits, "rock_cover_pct": round(cover, 2)}


def _haul_routes(t: Terrain, yard, plant) -> dict:
    from app.catalog import TBMS, VEHICLES
    from app.models import RouteRequest
    from app.routing import plan_routes

    req = RouteRequest(**{"from": list(yard), "to": list(plant), "equipment": ["hauler"],
                          "modes": ["accessible", "shortest"]})
    out = plan_routes(t, req, [VEHICLES["hauler"]], TBMS["tbm-4m"])
    routes = {}
    for name in ("shortest", "accessible"):
        r = out.get(name, {})
        if r.get("status") == "ok":
            co = r["geometry"]["coordinates"]
            routes[name] = [list(yard)] + co + [list(plant)]
    if not routes:
        raise ValueError("no haul route between the ore yard and the plant")
    return routes


@lru_cache(maxsize=16)
def _build(key: str) -> Site:
    from .model import Params
    cfg = json.loads(key)
    p = Params(**cfg["coeffs"])
    site = cfg["site"]
    t = get_terrain()
    for name, pt in [("plant", site["plant"]), ("ore", site["ore"]), ("pad", site["pad"]),
                     *[(f"portal {k}", v) for k, v in site["portals"].items()]]:
        if not t.contains(*pt):
            raise ValueError(f"{name} {pt} is outside the map")
    tiles = _Tiles(t, p)
    tunnels = {k: _tunnel(tiles, v, site["ore"]) for k, v in site["portals"].items()}
    ports = list(site["portals"].values())
    yard = [sum(x[0] for x in ports) / len(ports), sum(x[1] for x in ports) / len(ports)]
    routes = {k: _route(tiles, co) for k, co in _haul_routes(t, yard, site["plant"]).items()}
    half = site.get("pad_side_m", 400.0) / 2
    pad_tiles = []
    for e in np.linspace(-half, half, 5):
        for n in np.linspace(-half, half, 5):
            ti = tiles.at(*_offset(*site["pad"], e, n))
            if ti not in pad_tiles:
                pad_tiles.append(ti)
    return Site(tiles.list, tunnels, routes, {"center": list(site["pad"]), "tiles": pad_tiles},
                list(site["plant"]), yard)


COEFFS = ("wheel_damage_per_km_pct", "wheel_slope_ref_deg", "hard_bands_per_km", "dozer_damage_per_1000m3_pct")


def site_for(p) -> Site:
    key = json.dumps({"site": p.site, "coeffs": {k: getattr(p, k) for k in COEFFS}}, sort_keys=True)
    return _build(key)


def site_view(site: Site) -> dict:
    """The layout with per-tile odds, for the map."""
    def tile(ti):
        return site.tiles[ti]
    return {
        "plant": site.plant, "ore_yard": site.yard,
        "pad": {"center": site.pad["center"], "tiles": [tile(ti) for ti in site.pad["tiles"]]},
        "tunnels": {k: {"portal": v["portal"], "ore_target": v["ore"], "access_m": v["access_m"],
                        "segments": [{"from_m": a, "to_m": b, "tile": tile(ti)} for a, b, ti in v["segs"]]}
                    for k, v in site.tunnels.items()},
        "routes": {k: {"km": v["km"], "rock_cover_pct": v["rock_cover_pct"],
                       "wheel_hits_per_round_trip": round(v["hits_per_trip"], 5),
                       "geometry": {"type": "LineString", "coordinates": v["coords"]},
                       "segments": [{"km": d, "tile": tile(ti)} for d, ti in v["segs"]]}
                   for k, v in site.routes.items()},
        "basis": BASIS,
    }
