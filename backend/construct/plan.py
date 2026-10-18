"""Work plans on the map: which tiles a job covers, how long each takes, how risky it is.

A job is either a route (a road graded along a planned path) or a base (a square area
flattened around a centre). Each 400 m tile of the working grid that the job touches gets:

  work      unit-sols of excavation on that tile: the job's share of the tile, scaled up by
            slope (steeper ground means more cut and fill) and by hazard (rocks to clear,
            careful work near crater rims)
  hazard    0-1, from rock cover, roughness and nearness to crater rims
  incident_rate  incidents per unit-sol of work on the tile
  kinds     relative odds of each incident kind there (boulders on rocky tiles, slope
            failures on steep ones)
"""
from __future__ import annotations

import math

import numpy as np

from app.terrain import DEG_M, get_terrain, rock_cover_fraction

from .model import Params


def tile_hazard(t, r: int, c: int) -> dict:
    cover = float(rock_cover_fraction(t.rock_k[r, c], 0.25)) * 100      # % ground under rocks >= 25 cm
    rough = float(t.rough[r, c]) / t.rough_p95
    rim = float(t.rim_dist[r, c]) if np.isfinite(t.rim_dist[r, c]) else 1e9
    hazard = min(1.0, 0.5 * cover / 5 + 0.3 * min(rough, 1.5) + 0.4 * math.exp(-rim / 1500))
    return {"slope_deg": round(float(t.slope[r, c]), 2), "rock_cover_pct": round(cover, 2),
            "roughness": round(rough, 2), "crater_rim_m": round(min(rim, 99_999)), "hazard": round(hazard, 3)}


def _tile(t, p: Params, r: int, c: int, base_work: float, kind: str) -> dict:
    h = tile_hazard(t, r, c)
    s = h["slope_deg"]
    slope_f = 1 + (s / 2.0 if kind == "base" else (s / 4.0) ** 1.5)    # flattening a pad is more slope-sensitive
    hazard_f = 1 + 2 * h["hazard"]
    rate = p.incident_rate_per_unit_sol * (0.3 + 2.5 * h["hazard"]) * (1 + s / 10)
    kinds = {
        "boulder": 0.2 + h["rock_cover_pct"] / 2,
        "slope_failure": 0.05 + (s / 6) ** 2,
        "breakdown": 0.6,
        "stuck": 0.3 + 0.4 * max(0.0, 1 - h["roughness"]),               # smooth ground here is often loose sand
    }
    tot = sum(kinds.values())
    return {"r": int(r), "c": int(c), "lon": round(float(t.lon[c]), 5), "lat": round(float(t.lat[r]), 5), **h,
            "base_work": round(base_work, 3), "slope_factor": round(slope_f, 2), "hazard_factor": round(hazard_f, 2),
            "work": round(base_work * slope_f * hazard_f, 3), "incident_rate": round(rate, 4),
            "kinds": {k: round(v / tot, 3) for k, v in kinds.items()}}


def _offset(lon, lat, e, n):
    return lon + e / (DEG_M * math.cos(math.radians(lat))), lat + n / DEG_M


def route_plan(p: Params, frm, to, mode: str) -> dict:
    from app.catalog import TBMS, VEHICLES
    from app.models import RouteRequest
    from app.routing import plan_routes

    t = get_terrain()
    req = RouteRequest(**{"from": list(frm), "to": list(to), "equipment": ["utility-rover"], "modes": [mode]})
    out = plan_routes(t, req, [VEHICLES["utility-rover"]], TBMS["tbm-4m"])[mode]
    if out.get("status") != "ok":
        raise ValueError(f"no {mode} route: {out.get('reason', 'unknown')}")
    coords = [list(frm)] + out["geometry"]["coordinates"] + [list(to)]
    order, length = [], {}
    for a, b in zip(coords[:-1], coords[1:]):
        latm = math.radians((a[1] + b[1]) / 2)
        L = math.hypot((b[0] - a[0]) * DEG_M * math.cos(latm), (b[1] - a[1]) * DEG_M)
        n = max(1, int(L / 20))
        for j in range(n):
            f = (j + 0.5) / n
            rc = t.rc(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            if rc not in length:
                order.append(rc)
                length[rc] = 0.0
            length[rc] += L / n
    tiles = [_tile(t, p, r, c, length[(r, c)] / p.road_m_per_unit_sol, "route") for r, c in order]
    for tl, rc in zip(tiles, order):
        tl["length_m"] = round(length[rc])
    return {"kind": "route", "mode": mode, "geometry": {"type": "LineString", "coordinates": coords},
            "length_km": round(sum(length.values()) / 1000, 2), "tiles": tiles}


def base_plan(p: Params, center, area_m2: float) -> dict:
    t = get_terrain()
    side = math.sqrt(area_m2)
    n = max(4, int(side / 25))
    share: dict = {}
    order = []
    # serpentine sweep across the square, so units move to neighbouring tiles
    for i in range(n):
        cols = range(n) if i % 2 == 0 else range(n - 1, -1, -1)
        for j in cols:
            e = (j + 0.5) / n * side - side / 2
            nn = side / 2 - (i + 0.5) / n * side
            rc = t.rc(*_offset(center[0], center[1], e, nn))
            if rc not in share:
                order.append(rc)
                share[rc] = 0
            share[rc] += 1
    cell = area_m2 / (n * n)
    tiles = [_tile(t, p, r, c, share[(r, c)] * cell / p.area_m2_per_unit_sol, "base") for r, c in order]
    for tl, rc in zip(tiles, order):
        tl["area_m2"] = round(share[rc] * cell)
    half = side / 2
    ring = [_offset(center[0], center[1], e, nn) for e, nn in ((-half, half), (half, half), (half, -half), (-half, -half), (-half, half))]
    return {"kind": "base", "center": list(center), "area_m2": area_m2, "side_m": round(side),
            "geometry": {"type": "Polygon", "coordinates": [[[round(x, 5), round(y, 5)] for x, y in ring]]},
            "tiles": tiles}
