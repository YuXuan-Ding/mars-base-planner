"""Tunnel geometry, boring time and spoil for TBMs.

A tunnel is a straight line in plan. Its depth profile is the deepest one the TBM's
grade limit allows from each portal, so rock cover is as large as possible. The
tunnel is infeasible if that profile would break out to the surface (for example
under a valley).
"""
from __future__ import annotations

import math

import numpy as np

from .models import TBM, TunnelRequest
from .terrain import Terrain

SAMPLE_M = 100.0
DAYLIGHT_TOL_M = 1.0


def _line(t: Terrain, a: tuple[float, float], b: tuple[float, float]):
    L = t.distance_m(*a, *b)
    n = max(2, int(math.ceil(L / SAMPLE_M)) + 1)
    f = np.linspace(0, 1, n)
    lons = a[0] + (b[0] - a[0]) * f
    lats = a[1] + (b[1] - a[1]) * f
    return lons, lats, f * L, t.surface(lons, lats)


def _check(s, surf, z, tbm: TBM) -> dict:
    """Breakout and shallow-cover checks. z is the tunnel invert (floor) elevation."""
    crown_cover = surf - (z + tbm.diameter_m)
    out = surf - z < -DAYLIGHT_TOL_M
    out[0] = out[-1] = False
    breakout_at = float(s[np.argmax(out)]) if out.any() else None
    shallow = crown_cover < tbm.min_cover_m
    step = s[1] - s[0] if len(s) > 1 else 0.0
    # shallow runs touching each portal are the open approach cuts; shallow stretches
    # elsewhere are weak spots that need cut-and-cover or a deeper line
    n = len(s)
    head = int(np.argmin(shallow)) if not shallow.all() else n
    tail = int(np.argmin(shallow[::-1])) if not shallow.all() else 0
    mid = np.zeros(n, bool)
    mid[head: n - tail] = True
    bored = crown_cover[mid]
    return {
        "feasible": breakout_at is None,
        "breakout_at_m": breakout_at,
        "portal_cut_m": round(float((head + tail) * step)),
        "shallow_m": round(float((shallow & mid).sum() * step)),
        "min_cover_m": round(float(bored.min()), 1) if bored.size else 0.0,
        "max_cover_m": round(float(crown_cover.max()), 1),
    }


def transit_profile(t: Terrain, a, b, tbm: TBM) -> dict:
    lons, lats, s, surf = _line(t, a, b)
    g = tbm.max_grade_pct / 100
    L = s[-1]
    floor = surf.min() - tbm.min_cover_m - tbm.diameter_m
    z = np.maximum.reduce([surf[0] - g * s, surf[-1] - g * (L - s), np.full_like(s, floor)])
    return {"lons": lons, "lats": lats, "s": s, "surf": surf, "z": z, "length_m": float(L), **_check(s, surf, z, tbm)}


def climb_profile(t: Terrain, a, b, tbm: TBM, max_grade: float) -> dict:
    """Bore up a scarp: from the low portal the tunnel climbs at grade while keeping its cover, then a
    spiral under the high portal gains whatever height is left (like a railway spiral tunnel)."""
    lons, lats, s, surf = _line(t, a, b)
    if surf[0] > surf[-1]:
        p = climb_profile(t, b, a, tbm, max_grade)
        return {**p, **{k: p[k][::-1] for k in ("lons", "lats", "surf", "z")}, "s": p["s"][-1] - p["s"][::-1]}
    cap = surf - tbm.diameter_m - tbm.min_cover_m
    cap[0] = surf[0]
    suffix = np.minimum.accumulate(cap[::-1])[::-1]       # never rise above the cover line further on
    z = np.minimum(surf[0] + max_grade * s, suffix)
    z[0] = surf[0]
    spiral = max(0.0, float(surf[-1] - z[-1])) / max_grade
    return {"lons": lons, "lats": lats, "s": s, "surf": surf, "z": z,
            "length_m": float(s[-1]) + spiral, "spiral_m": round(spiral), **_check(s, surf, z, tbm)}


def vertical_m(p: dict) -> float:
    """Height travelled inside a bore, up and down alike: the rises and falls of its floor, plus any height a
    spiral or portal ramp makes up between the floor's ends and the ground at the portals."""
    z, surf = p["z"], p["surf"]
    return float(np.abs(np.diff(z)).sum() + abs(surf[0] - z[0]) + abs(surf[-1] - z[-1]))


def decline_profile(t: Terrain, a, b, depth_m: float, tbm: TBM) -> dict:
    """Portal at a, descending at max grade toward a point depth_m under b.
    If the straight run is too short to get that deep, a spiral decline at b makes up the rest."""
    lons, lats, s, surf = _line(t, a, b)
    g = tbm.max_grade_pct / 100
    z_target = surf[-1] - depth_m
    z = np.maximum(surf[0] - g * s, z_target)
    decline = max(0.0, float(z[-1] - z_target)) / g
    return {"lons": lons, "lats": lats, "s": s, "surf": surf, "z": z, "length_m": float(s[-1]),
            "decline_m": round(decline), "target_elev_m": round(float(z_target), 1), **_check(s, surf, z, tbm)}


def find_portal(t: Terrain, lon: float, lat: float, radius_km: float, max_slope: float):
    """Flattest nearby cell for a portal pad, trading slope against distance."""
    r, c = t.rc(lon, lat)
    R, C = t.shape
    nr = int(math.ceil(radius_km * 1000 / t.dy))
    nc = int(math.ceil(radius_km * 1000 / t.dx[r]))
    r0, r1, c0, c1 = max(r - nr, 0), min(r + nr + 1, R), max(c - nc, 0), min(c + nc + 1, C)
    sl = t.slope[r0:r1, c0:c1]
    yy, xx = np.mgrid[r0:r1, c0:c1]
    d = np.hypot((yy - r) * t.dy, (xx - c) * t.dx[r])
    ok = (sl <= max_slope) & (d <= radius_km * 1000)
    if not ok.any():
        return None
    score = np.where(ok, sl / max_slope + d / (radius_km * 1000), np.inf)
    i = np.unravel_index(np.argmin(score), score.shape)
    pr, pc = int(i[0] + r0), int(i[1] + c0)
    plon, plat = t.lonlat(pr, pc)
    return {"lon": float(plon), "lat": float(plat), "slope_deg": round(float(t.slope[pr, pc]), 2),
            "distance_m": round(float(d[i]))}


def bore_rate(tbm: TBM) -> float:
    return tbm.advance_m_per_sol * tbm.duty


def face_area(tbm: TBM) -> float:
    return math.pi * (tbm.diameter_m / 2) ** 2


def path3d(p: dict, max_pts: int = 200) -> list[list[float]]:
    n = len(p["s"])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_pts)).round().astype(int))
    return [[round(float(p["lons"][i]), 5), round(float(p["lats"][i]), 5), round(float(p["z"][i]), 1)] for i in idx]


def profile(p: dict, max_pts: int = 200) -> list[list[float]]:
    n = len(p["s"])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_pts)).round().astype(int))
    return [[round(float(p["s"][i]) / 1000, 3), round(float(p["surf"][i]), 1), round(float(p["z"][i]), 1)] for i in idx]


def plan(t: Terrain, req: TunnelRequest, tbm: TBM) -> dict:
    count = req.machine.count
    rate = bore_rate(tbm)
    area = face_area(tbm)
    assumptions = [
        f"advance {tbm.advance_m_per_sol} m/sol at {tbm.duty:.0%} duty in average rock; no subsurface data, rock hardness unknown",
        f"spoil swell factor {tbm.swell}",
    ]

    portal = find_portal(t, *req.from_, req.portal_search_radius_km, req.portal_max_slope_deg)
    if portal is None:
        return {"status": "no_portal", "reason": f"no ground under {req.portal_max_slope_deg}° within "
                f"{req.portal_search_radius_km} km of the start", "assumptions": assumptions}
    a = (portal["lon"], portal["lat"])
    portals = [portal]

    if req.purpose == "transit":
        far = find_portal(t, *req.to, req.portal_search_radius_km, req.portal_max_slope_deg)
        if far is None:
            return {"status": "no_portal", "reason": "no flat ground for the far portal", "assumptions": assumptions}
        portals.append(far)
        p = transit_profile(t, a, (far["lon"], far["lat"]), tbm)
        main_m, extra_m, faces = p["length_m"], 0.0, min(count, 2)
        sols = main_m / (rate * faces)
        spoil_solid = area * main_m
        result = {"faces": faces}
    else:
        p = decline_profile(t, a, tuple(req.to), req.depth_m, tbm)
        main_m = p["length_m"] + p["decline_m"]
        if req.purpose == "mining":
            extra_m = req.drift_length_m
            ore_t = area * extra_m * req.ore_fraction * req.rock_density_t_m3
            spoil_solid = area * (main_m + extra_m * (1 - req.ore_fraction))
            result = {
                "drift_length_m": round(extra_m),
                "ore_t": round(ore_t),
                "ore_t_per_sol": round(area * rate * count * req.ore_fraction * req.rock_density_t_m3, 1),
            }
            assumptions.append(f"ore fraction {req.ore_fraction:.0%}, rock density {req.rock_density_t_m3} t/m3 (assumed)")
        else:
            extra_m = req.chamber_length_m
            spoil_solid = area * (main_m + extra_m)
            result = {"chamber_length_m": round(extra_m), "chamber_volume_m3": round(area * extra_m)}
        # one machine drives the access tunnel; all of them work the drifts/chamber
        sols = main_m / rate + extra_m / (rate * count)
        result["decline_m"] = p["decline_m"]
        result["target_elev_m"] = p["target_elev_m"]

    status = "ok" if p["feasible"] else "breakout"
    out = {
        "status": status,
        "purpose": req.purpose,
        "machine": {"model": tbm.id, "count": count},
        "portals": portals,
        "length_m": round(main_m + extra_m),
        "access_length_m": round(main_m),
        "sols": round(sols, 1),
        "spoil_m3": round(spoil_solid * tbm.swell),
        "spoil_m3_per_machine_sol": round(area * rate * tbm.swell, 1),
        "cover": {k: p[k] for k in ("min_cover_m", "max_cover_m", "shallow_m", "portal_cut_m")},
        "path3d": path3d(p),
        "profile": profile(p),
        "assumptions": assumptions,
        **result,
    }
    if status == "breakout":
        out["reason"] = f"tunnel would reach the surface {p['breakout_at_m'] / 1000:.1f} km from the portal"
    return out
