"""Base site search: where in an area does a set of facilities need the least flattening?

Every facility is a square pad of its area, on the native 200 m grid. At every position we fit a plane
to the pad with summed-area tables (constant time per position), let the pad keep the tilt it is
allowed, and estimate earthwork (cut + fill, which balance, so each is about half):

    V = A * 0.8 * sigma  +  max(0, |grad| - tan(max_slope)) * s^3 / 4

sigma is the bumpiness left after the best plane (0.8 sigma is the mean |residual| for normal noise),
scaled down for pads smaller than the fit window; the second term levels the tilt beyond the
allowance (mean |height| over a tilted square of side s is slope * s / 4). The fit window is at
least 3 x 3 cells (600 m) so curvature shows up, which makes this a screening estimate for small
pads; HiRISE elevation at the chosen site would refine it. Where earthwork ties (flat enough
ground costs nothing), the flatter spot wins.

The base is then laid out greedily around each candidate centre: largest pad first, cheapest free spot
within the spread radius, no overlaps, landing pads kept their separation from everything. The centre
with the least total earthwork wins.
"""
from __future__ import annotations

import math

import numpy as np
from fastapi import HTTPException
from scipy import ndimage

from .models import Facility, SiteRequest
from .terrain import DEG_M, Terrain

GAP_M = 30.0              # clear strip between neighbouring pads
MAX_CANDIDATES = 600      # base centres scored per search
MAX_AREA_CELLS = 900_000  # about 190 x 190 km at 200 m
TRACE_MAP = 90            # cost maps are sent at most TRACE_MAP x TRACE_MAP


def _box(a: np.ndarray, k: int) -> np.ndarray:
    """Sums over every k x k window; result[i, j] covers a[i:i+k, j:j+k]."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]


def _cost_map(H: np.ndarray, f: Facility, res_x: float, res_y: float) -> dict:
    """Earthwork, pad level and fitted tilt for a pad of facility f centred on every cell."""
    side = math.sqrt(f.area_m2)
    k = max(3, int(round(side / min(res_x, res_y))))
    R, C = H.shape
    out = {"side_m": side, "k": k, "V": np.full((R, C), np.inf), "level": np.full((R, C), np.nan),
           "tilt": np.full((R, C), np.nan), "sigma": np.full((R, C), np.nan)}
    if k > R or k > C:
        return out
    yy, xx = np.indices(H.shape, dtype=np.float64)
    n = k * k
    Sh, Shh = _box(H, k), _box(H * H, k)
    Sxh, Syh = _box(xx * H, k), _box(yy * H, k)
    x0 = np.arange(C - k + 1)[None, :] + (k - 1) / 2
    y0 = np.arange(R - k + 1)[:, None] + (k - 1) / 2
    sxx = k * k * (k * k - 1) / 12.0                    # sum of squared centred offsets along one axis
    bx = (Sxh - x0 * Sh) / sxx                          # m per cell, east
    by = (Syh - y0 * Sh) / sxx                          # m per cell, south
    var = (Shh - Sh * Sh / n - bx * bx * sxx - by * by * sxx) / n
    sigma = np.sqrt(np.clip(var, 0, None)) * min(1.0, side / (k * min(res_x, res_y)))
    grad = np.hypot(bx / res_x, by / res_y)
    excess = np.clip(grad - math.tan(math.radians(f.max_slope_deg)), 0, None)
    V = f.area_m2 * 0.8 * sigma + excess * side ** 3 / 4
    o = k // 2                                           # window top-left -> centre cell
    s = (slice(o, o + R - k + 1), slice(o, o + C - k + 1))
    out["V"][s], out["level"][s] = V, Sh / n
    out["tilt"][s], out["sigma"][s] = np.degrees(np.arctan(grad)), sigma
    return out


def _inside(poly: np.ndarray, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for a lon x lat grid of cell centres (rows = lats)."""
    X, Y = np.meshgrid(lons, lats)
    inside = np.zeros(X.shape, bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        if ya == yb:
            continue
        crosses = (Y >= min(ya, yb)) & (Y < max(ya, yb))
        xi = xa + (Y - ya) * (xb - xa) / (yb - ya)
        inside ^= crosses & (X < xi)
    return inside


def _items(req: SiteRequest, catalog: dict[str, Facility]) -> list[Facility]:
    items = []
    for ch in req.facilities:
        if ch.type not in catalog:
            raise HTTPException(422, f"unknown facility '{ch.type}'; options: {sorted(catalog)}")
        base = catalog[ch.type]
        f = base.model_copy(update={k: v for k, v in (("area_m2", ch.area_m2), ("max_slope_deg", ch.max_slope_deg)) if v})
        items += [f] * ch.count
    return sorted(items, key=lambda f: -f.area_m2)        # largest pads are placed first


def search(t: Terrain, req: SiteRequest, catalog: dict[str, Facility]) -> dict:
    m = t.meta
    poly = np.array(req.polygon, float) if req.polygon else None
    if poly is not None:
        (lo0, la0), (lo1, la1) = poly.min(0), poly.max(0)
    elif req.area:
        (lo0, la0), (lo1, la1) = req.area
    else:
        raise HTTPException(422, "send either area (two corners) or polygon (an outline)")
    west, east, south, north = min(lo0, lo1), max(lo0, lo1), min(la0, la1), max(la0, la1)
    res = t.native_res
    r0, r1 = int((m["lat_north"] - north) / res), int(math.ceil((m["lat_north"] - south) / res))
    c0, c1 = int((west - m["lon_west"]) / res), int(math.ceil((east - m["lon_west"]) / res))
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, t.native.shape[0]), min(c1, t.native.shape[1])
    if (r1 - r0) * (c1 - c0) > MAX_AREA_CELLS:
        raise HTTPException(422, "search area is too large; keep it under about 190 x 190 km")
    H = t.native[r0:r1, c0:c1].astype(np.float64)
    R, C = H.shape
    lat_c = m["lat_north"] - (r0 + R / 2) * res
    res_y, res_x = res * DEG_M, res * DEG_M * math.cos(math.radians(lat_c))
    items = _items(req, catalog)
    if R < 3 or C < 3:
        raise HTTPException(422, "search area is too small; it needs at least 3 x 3 cells of 200 m")

    # cells inside a drawn outline; a pad only counts where its whole square fits inside
    inside = None
    if poly is not None:
        lons = m["lon_west"] + (c0 + np.arange(C) + 0.5) * res
        lats = m["lat_north"] - (r0 + np.arange(R) + 0.5) * res
        inside = _inside(poly, lons, lats)
        if inside.sum() < 9:
            raise HTTPException(422, "search area is too small; it needs at least 3 x 3 cells of 200 m")

    # 1. a cost map per facility type
    types = {f.id + f"@{f.area_m2:.0f}/{f.max_slope_deg}": f for f in items}
    maps = {key: _cost_map(H, f, res_x, res_y) for key, f in types.items()}
    if inside is not None:
        for key, f in types.items():
            half = math.sqrt(f.area_m2) / 2
            ry, rx = int(math.floor(half / res_y)), int(math.floor(half / res_x))
            fits = ndimage.binary_erosion(inside, np.ones((2 * ry + 1, 2 * rx + 1), bool), border_value=0) if ry or rx else inside
            maps[key]["V"][~fits] = np.inf
    key_of = lambda f: f.id + f"@{f.area_m2:.0f}/{f.max_slope_deg}"

    # 2. score base centres with a greedy layout around each
    stride = max(1, math.ceil(math.sqrt(R * C / MAX_CANDIDATES)))
    centres = [(r, c) for r in range(stride // 2, R, stride) for c in range(stride // 2, C, stride)]
    if inside is not None:
        # same density inside an outline as a full rectangle of that size would get
        stride = max(1, math.ceil(math.sqrt(inside.sum() / MAX_CANDIDATES)))
        centres = [(r, c) for r in range(stride // 2, R, stride) for c in range(stride // 2, C, stride) if inside[r, c]]
    spread = req.max_spread_km * 1000
    reach = max(spread + f.min_separation_m for f in items)
    wr, wc = int(math.ceil(reach / res_y)), int(math.ceil(reach / res_x))
    oy, ox = np.mgrid[-wr: wr + 1, -wc: wc + 1]
    dist = np.hypot(oy * res_y, ox * res_x)

    def layout(rc: int, cc: int, log: list | None = None):
        placed, total, tie = [], 0.0, 0.0
        for f in items:
            cm = maps[key_of(f)]
            ra, rb, ca, cb = rc - wr, rc + wr + 1, cc - wc, cc + wc + 1
            sub = np.full(dist.shape, np.inf)
            sr0, sc0 = max(ra, 0), max(ca, 0)
            sr1, sc1 = min(rb, R), min(cb, C)
            sub[sr0 - ra: sr1 - ra, sc0 - ca: sc1 - ca] = cm["V"][sr0:sr1, sc0:sc1]
            tilt = np.zeros(dist.shape)
            tilt[sr0 - ra: sr1 - ra, sc0 - ca: sc1 - ca] = np.nan_to_num(cm["tilt"][sr0:sr1, sc0:sc1])
            ok = dist <= spread + f.min_separation_m
            pr, pc = rc + oy, cc + ox
            for (qr, qc, g) in placed:
                dy_m, dx_m = np.abs(pr - qr) * res_y, np.abs(pc - qc) * res_x
                need = (math.sqrt(f.area_m2) + math.sqrt(g.area_m2)) / 2 + GAP_M
                ok &= (dy_m >= need) | (dx_m >= need)          # squares don't overlap
                sep = max(f.min_separation_m, g.min_separation_m)
                if sep:
                    ok &= np.hypot(dy_m, dx_m) >= sep
            cand = np.where(ok, sub, np.inf)
            # earthwork first; among equal earthwork, flatter ground, then closer to the centre
            i = int(np.argmin(cand + 1e-3 * tilt + 1e-6 * dist))
            v = float(cand.flat[i])
            if not math.isfinite(v):
                if log is not None:
                    log.append({"facility": f.id, "name": f.name, "options": 0})
                return None, placed
            qr, qc = int(pr.flat[i]), int(pc.flat[i])
            placed.append((qr, qc, f))
            total += v
            tie += float(tilt.flat[i])
            if log is not None:
                log.append({"facility": f.id, "name": f.name, "options": int(np.isfinite(cand).sum()),
                            "search_radius_m": round(spread + f.min_separation_m), "at": lonlat(qr, qc),
                            "earthwork_m3": round(v)})
        return (total, tie), placed

    def lonlat(r, c):
        return [round(m["lon_west"] + (c0 + c + 0.5) * res, 5), round(m["lat_north"] - (r0 + r + 0.5) * res, 5)]

    keys = [layout(r, c)[0] for (r, c) in centres]          # (earthwork, summed pad tilt) or None
    scores = [k[0] if k else None for k in keys]
    order = sorted((k, i) for i, k in enumerate(keys) if k is not None)
    if not order:
        return {"status": "none", "reason": "the facilities do not fit inside the area with their spacing; "
                "widen the area or the spread, or drop a landing pad"}

    # pick the best and a few alternatives that are clearly elsewhere
    chosen = []
    for s, i in order:
        r, c = centres[i]
        if all(math.hypot((r - centres[j][0]) * res_y, (c - centres[j][1]) * res_x) >= spread for _, j in chosen):
            chosen.append((s, i))
        if len(chosen) > req.alternatives:
            break

    def describe(i: int, log: list | None = None) -> dict:
        r, c = centres[i]
        (total, _), placed = layout(r, c, log)
        pads = []
        for (qr, qc, f) in placed:
            cm = maps[key_of(f)]
            v, side = float(cm["V"][qr, qc]), math.sqrt(f.area_m2)
            lon, lat = lonlat(qr, qc)
            dlat, dlon = side / 2 / DEG_M, side / 2 / (DEG_M * math.cos(math.radians(lat)))
            pads.append({
                "facility": f.id, "name": f.name, "center": [lon, lat], "side_m": round(side),
                "area_m2": round(f.area_m2), "max_slope_deg": f.max_slope_deg,
                "polygon": [[round(lon - dlon, 5), round(lat - dlat, 5)], [round(lon + dlon, 5), round(lat - dlat, 5)],
                            [round(lon + dlon, 5), round(lat + dlat, 5)], [round(lon - dlon, 5), round(lat + dlat, 5)]],
                "earthwork_m3": round(v), "cut_m3": round(v / 2), "fill_m3": round(v / 2),
                "pad_elevation_m": round(float(cm["level"][qr, qc])),
                "ground_slope_deg": round(float(cm["tilt"][qr, qc]), 2),
                "bumpiness_m": round(float(cm["sigma"][qr, qc]), 2),
                "grading_sols": round(v / req.grading_rate_m3_per_sol, 1),
            })
        return {"center": lonlat(r, c), "earthwork_m3": round(total), "flatten_area_m2": round(sum(f.area_m2 for f in items)),
                "grading_sols": round(total / req.grading_rate_m3_per_sol, 1), "facilities": pads}

    steps: list = []
    best = describe(chosen[0][1], steps if req.trace else None)
    out = {
        "status": "ok",
        "best": best,
        "alternatives": [describe(i) for _, i in chosen[1:]],
        "evaluated": {"positions_per_facility": int(R * C if inside is None else inside.sum()),
                      "area_km2": round(float((R * C if inside is None else inside.sum()) * res_x * res_y) / 1e6, 1), "base_centres": len(centres),
                      "feasible_centres": len(order), "facility_types": len(types), "grid_resolution_m": round(res_y)},
        "assumptions": [
            "pads are squares of the facility area on 200 m elevation; earthwork = bumpiness after the best-fit plane "
            "+ levelling the tilt beyond the pad's allowance; cut and fill balance",
            f"pads keep a {GAP_M:.0f} m clear strip between them; landing pads keep their separation from everything",
            f"grading at {req.grading_rate_m3_per_sol:g} m3 per sol",
        ],
    }
    if req.trace:
        out["trace"] = _trace(maps, types, centres, scores, steps, lonlat, R, C, order, chosen)
    return out


def _trace(maps, types, centres, scores, steps, lonlat, R, C, order, chosen) -> dict:
    """Cost maps (thinned to TRACE_MAP cells a side, keeping each block's best value, log-scaled 0-250,
    255 = no pad fits), every base centre in evaluation order with its score, and the layout steps."""
    b = max(1, math.ceil(max(R, C) / TRACE_MAP))
    rr, cc = math.ceil(R / b), math.ceil(C / b)
    cost_maps = []
    for key, f in types.items():
        V = maps[key]["V"]
        P = np.full((rr * b, cc * b), np.inf)
        P[:R, :C] = V
        blk = P.reshape(rr, b, cc, b).min(axis=(1, 3))
        fin = np.isfinite(blk)
        lo, hi = (float(blk[fin].min()), float(blk[fin].max())) if fin.any() else (0.0, 1.0)
        q = np.full(blk.shape, 255, dtype=int)
        if fin.any():
            ll, lh = math.log10(lo + 1), math.log10(hi + 1)
            q[fin] = np.round((np.log10(blk[fin] + 1) - ll) / max(lh - ll, 1e-9) * 250)
        cost_maps.append({"facility": f.id, "name": f.name, "side_m": round(math.sqrt(f.area_m2)),
                          "rows": rr, "cols": cc, "block_cells": b,
                          "nw": lonlat(-0.5, -0.5), "se": lonlat(rr * b - 0.5, cc * b - 0.5),
                          "min_m3": round(lo), "max_m3": round(hi), "values": q.ravel().tolist()})
    rank = {i: n for n, (_, i) in enumerate(order)}
    return {
        "cost_maps": cost_maps,
        "centres": [[*lonlat(r, c), None if s is None else round(s), rank.get(i)] for i, ((r, c), s) in enumerate(zip(centres, scores))],
        "chosen": [rank[i] for _, i in chosen],
        "layout_steps": steps,
    }


def patch(t: Terrain, lon: float, lat: float, half_km: float) -> dict:
    """Native 200 m elevation around a point, for detail rendering when zoomed in."""
    m, res = t.meta, t.native_res
    dlat = half_km * 1000 / DEG_M
    dlon = half_km * 1000 / (DEG_M * math.cos(math.radians(lat)))
    r0 = max(int((m["lat_north"] - (lat + dlat)) / res), 0)
    r1 = min(int(math.ceil((m["lat_north"] - (lat - dlat)) / res)), t.native.shape[0])
    c0 = max(int((lon - dlon - m["lon_west"]) / res), 0)
    c1 = min(int(math.ceil((lon + dlon - m["lon_west"]) / res)), t.native.shape[1])
    H = t.native[r0:r1, c0:c1]
    return {"rows": int(H.shape[0]), "cols": int(H.shape[1]),
            "lon_west": m["lon_west"] + c0 * res, "lon_east": m["lon_west"] + c1 * res,
            "lat_north": m["lat_north"] - r0 * res, "lat_south": m["lat_north"] - r1 * res,
            "resolution_m": round(res * DEG_M), "h": H.round().astype("<i2")}
