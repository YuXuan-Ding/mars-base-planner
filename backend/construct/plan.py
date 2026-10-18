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


def _m(a, b) -> float:
    latm = math.radians((a[1] + b[1]) / 2)
    return math.hypot((b[0] - a[0]) * DEG_M * math.cos(latm), (b[1] - a[1]) * DEG_M)


TOUCH_M = 600.0   # a road within this of another road's end, or of a portal, connects to it


def _groups(legs: list, portals: list) -> list[int]:
    """Which stretch of road each leg belongs to. Roads that touch are one stretch; only bores split the
    road, so a stretch that touches no portal (a gap the fleet drives across, an internal base road) joins
    the nearest stretch."""
    n = len(legs)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    near = lambda A, B: min(_m(a, b) for a in A for b in B)
    for i in range(n):
        for j in range(i + 1, n):
            ends_i, ends_j = [legs[i][0], legs[i][-1]], [legs[j][0], legs[j][-1]]
            if min(near(ends_i, legs[j]), near(ends_j, legs[i])) <= TOUCH_M:
                par[find(i)] = find(j)
    touches = lambda r: any(near([q], legs[i]) <= TOUCH_M for q in portals for i in range(n) if find(i) == r)
    while True:
        roots = sorted({find(i) for i in range(n)})
        loose = [r for r in roots if not touches(r)] if portals else roots[1:]
        if len(roots) < 2 or not loose:
            break
        r = loose[0]
        mine = [i for i in range(n) if find(i) == r]
        other = min((i for i in range(n) if find(i) != r), key=lambda i: min(near(legs[k], legs[i]) for k in mine))
        par[r] = find(other)
    roots = sorted({find(i) for i in range(n)})
    return [roots.index(find(i)) for i in range(n)]


def route_plan(p: Params, frm, to, mode: str, paths=None, bores=None) -> dict:
    """A road along `paths` (the route the user picked: one or more [lon, lat] polylines, e.g. the
    two surface legs of a tunnel route), or else along a route planned here for a utility rover.
    `bores` ([{coordinates: [portal, ..., portal], length_m}]) are bored by TBMs during the job: a portal
    set-up tile at each end, then tiles along the line; road crews can't cross a bore until it breaks
    through."""
    t = get_terrain()
    if paths is not None or bores:
        legs = [[list(map(float, q)) for q in leg] for leg in (paths or []) if len(leg) > 1]
        if not legs and not bores:
            raise ValueError("paths has no line with two or more points")
    else:
        from app.catalog import TBMS, VEHICLES
        from app.models import RouteRequest
        from app.routing import plan_routes

        req = RouteRequest(**{"from": list(frm), "to": list(to), "equipment": ["utility-rover"], "modes": [mode]})
        out = plan_routes(t, req, [VEHICLES["utility-rover"]], TBMS["tbm-4m"])[mode]
        if out.get("status") != "ok":
            raise ValueError(f"no {mode} route: {out.get('reason', 'unknown')}")
        if "geometry" in out:
            legs = [[list(frm)] + out["geometry"]["coordinates"] + [list(to)]]
        else:   # tunnel and bore-through routes: only their roads are graded
            legs = [l["geometry"]["coordinates"] for l in out.get("legs", out.get("surface_legs", []))]
            if not legs:
                raise ValueError(f"the {mode} route has no road to grade")
    bores = [b for b in (bores or []) if len(b.get("coordinates", [])) > 1]
    portals = [q for b in bores for q in (b["coordinates"][0], b["coordinates"][-1])]
    group_of = _groups(legs, portals)
    order, length, grp = [], {}, {}
    for li, leg in enumerate(legs):
        for a, b in zip(leg[:-1], leg[1:]):
            L = _m(a, b)
            n = max(1, int(L / 20))
            for j in range(n):
                f = (j + 0.5) / n
                rc = t.rc(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                if rc not in length:
                    order.append(rc)
                    length[rc], grp[rc] = 0.0, group_of[li]
                length[rc] += L / n
    tiles = [_tile(t, p, r, c, length[(r, c)] / p.road_m_per_unit_sol, "route") for r, c in order]
    for tl, rc in zip(tiles, order):
        tl.update(kind="road", length_m=round(length[rc]), group=grp[rc])

    # bores: portal set-up at each end, then the line cut into the 400 m cells it crosses, in order A -> B
    bore_plans = []
    for bi, b in enumerate(bores):
        line = [list(map(float, q)) for q in b["coordinates"]]
        line_m = sum(_m(a, c) for a, c in zip(line[:-1], line[1:])) or 1.0
        scale = float(b.get("length_m") or line_m) / line_m        # spirals and ramps make a bore longer than its line
        seq, blen = [], {}
        for a, c in zip(line[:-1], line[1:]):
            L = _m(a, c)
            n = max(1, int(L / 20))
            for j in range(n):
                f = (j + 0.5) / n
                rc = t.rc(a[0] + (c[0] - a[0]) * f, a[1] + (c[1] - a[1]) * f)
                if rc not in blen:
                    seq.append(rc)
                    blen[rc] = 0.0
                blen[rc] += L / n * scale

        def bore_tile(rc, kind, work, extra):
            r, c = rc
            h = tile_hazard(t, r, c)
            tiles.append({"r": int(r), "c": int(c), "lon": round(float(t.lon[c]), 5), "lat": round(float(t.lat[r]), 5),
                          **h, "base_work": round(work, 3), "slope_factor": 1.0, "hazard_factor": 1.0,
                          "work": round(work, 3), "kind": kind, "bore": bi,
                          "incident_rate": round(p.tbm_incident_rate_per_sol, 4) if kind == "bore" else 0.0,
                          "kinds": {"breakdown": 1.0, "boulder": 0.0, "slope_failure": 0.0, "stuck": 0.0}, **extra})
            return len(tiles) - 1
        pa = bore_tile(t.rc(*line[0]), "portal", p.bore_setup_sols, {"length_m": 0})
        cut = [bore_tile(rc, "bore", blen[rc] / p.bore_m_per_tbm_sol, {"length_m": round(blen[rc])}) for rc in seq]
        pb = bore_tile(t.rc(*line[-1]), "portal", p.bore_setup_sols, {"length_m": 0})
        side = []
        for q in (line[0], line[-1]):
            d = [(min(_m(q, v) for v in leg), group_of[li]) for li, leg in enumerate(legs)]
            dm, g = min(d)
            side.append(g if dm <= TOUCH_M else None)
        bore_plans.append({"tiles": cut, "portal_a": pa, "portal_b": pb, "groups": side,
                           "length_m": round(sum(blen.values())), "coordinates": line})
    geom = {"type": "LineString", "coordinates": legs[0]} if len(legs) == 1 else {"type": "MultiLineString", "coordinates": legs}
    return {"kind": "route", "mode": mode, "geometry": geom,
            "length_km": round(sum(length.values()) / 1000, 2),
            "bore_km": round(sum(b["length_m"] for b in bore_plans) / 1000, 2), "bores": bore_plans,
            "groups": len(set(group_of)), "tiles": tiles}


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
