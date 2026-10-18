"""Surface and tunnel routing on the working grid.

The grid is a 16-connected graph (king + knight moves), so paths are not limited to
45° turns. Grade is measured along each move, not from cell slope, so switchbacks
emerge naturally. Three route kinds:

- accessible: drivable today. Moves that break any fleet hazard limit are removed.
- shortest:   everything is allowed except cliffs; steep or rocky moves cost earthwork.
- tunnel:     accessible surface legs joined by at most one straight TBM tunnel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from . import tunnels
from .models import TBM, HazardLimits, HazardWeights, RouteRequest, Vehicle
from .terrain import Grid, Terrain, rock_cover_fraction

# Half of the 16 neighbours (the graph is undirected). Knight moves also list the two
# cells they pass between, which must be passable too.
OFFSETS = [
    ((0, 1), None), ((1, 0), None), ((1, 1), None), ((1, -1), None),
    ((1, 2), ((0, 1), (1, 1))), ((2, 1), ((1, 0), (1, 1))),
    ((2, -1), ((1, 0), (1, -1))), ((1, -2), ((0, -1), (1, -1))),
]
CLIFF_DEG = 35.0           # never build a road up a steeper move than this
MIN_ROCK_COVER = 0.001     # "0 %" in a request means below 0.1 % modeled cover
SHORTEST_AVOIDANCE = 0.3   # the shortest route still leans slightly away from hazards
PORTAL_MAX_SLOPE = 5.0
TWO_LANE_TRIPS = 24
MIN_TUNNEL_SAVING = 0.9   # with a drivable route, a tunnel must cut trip length by 10 %+        # more trips per sol than this gets a two-lane road


# fleet

@dataclass
class Fleet:
    limits: HazardLimits
    binding: dict[str, str]
    vehicles: list[Vehicle]
    road_width_m: float
    lanes: int
    hauler: Vehicle | None
    trips_per_sol: int
    slowest_kmh: float


def make_fleet(vehicles: list[Vehicle], req: RouteRequest) -> Fleet:
    """The fleet is held to its most restrictive member on every limit."""
    lim, binding = {}, {}
    for name in HazardLimits.model_fields:
        pick = max if name == "min_crater_rim_clearance_m" else min
        v = pick(vehicles, key=lambda v: getattr(v.limits, name))
        lim[name], binding[name] = getattr(v.limits, name), v.id
    if req.hazard_overrides:
        for name, val in req.hazard_overrides.model_dump(exclude_none=True).items():
            lim[name], binding[name] = val, "override"

    carriers = [v for v in vehicles if v.hauls and v.payload_t > 0]
    hauler = max(carriers, key=lambda v: v.payload_t) if carriers else None
    trips = math.ceil(req.throughput_t_per_sol / hauler.payload_t) if hauler and req.throughput_t_per_sol else 0
    lanes = 2 if trips > TWO_LANE_TRIPS else 1
    width = (max(v.width_m for v in vehicles) + 1.0) * lanes
    return Fleet(HazardLimits(**lim), binding, vehicles, width, lanes, hauler, trips,
                 min(v.speed_kmh for v in vehicles))


# per-window graph

@dataclass
class Ctx:
    t: Terrain
    g: Grid
    fleet: Fleet
    weights: HazardWeights
    src: np.ndarray = field(init=False)
    dst: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    dh: np.ndarray = field(init=False)
    cross: np.ndarray = field(init=False)      # tan of sideways tilt across the move
    move_ok_cells: np.ndarray = field(init=False)

    def __post_init__(self):
        g, lim = self.g, self.fleet.limits
        R, C = g.shape
        self.n = R * C
        rock_limit = max(lim.max_rock_cover_pct / 100, MIN_ROCK_COVER)
        self.rock_frac = rock_cover_fraction(g.rock_k, lim.max_rock_m)
        self.rock_ok = self.rock_frac <= rock_limit
        self.rim_ok = g.rim_dist >= lim.min_crater_rim_clearance_m
        self.cell_ok = self.rock_ok & self.rim_ok

        w = self.weights
        terms = [
            (w.slope, np.clip(g.slope / lim.max_grade_deg, 0, 1)),
            (w.rock, np.clip(self.rock_frac / rock_limit, 0, 1)),
            (w.roughness, np.clip(g.rough / self.t.rough_p95, 0, 1)),
            (w.crater, np.exp(-g.rim_dist / 2000.0)),
        ]
        tot = sum(wt for wt, _ in terms) or 1.0
        self.hazard = sum(wt * a for wt, a in terms) / tot

        idx = np.arange(self.n).reshape(R, C)
        ok = self.cell_ok
        S, D, Ls, DH, CR, OK = [], [], [], [], [], []
        for (dr, dc), between in OFFSETS:
            if dr >= R or abs(dc) >= C:
                continue
            c0, c1 = max(0, -dc), C - max(0, dc)
            src = idx[: R - dr, c0:c1]
            dst = idx[dr:, c0 + dc: c1 + dc]
            m = ok[: R - dr, c0:c1] & ok[dr:, c0 + dc: c1 + dc]
            if between:
                for br, bc in between:
                    m = m & ok[br: R - dr + br, c0 + bc: c1 + bc]
            dxr = g.dx[: R - dr, None]
            ex, ey = dc * dxr, -dr * g.dy
            L = np.hypot(ex, ey) * np.ones_like(src, dtype=np.float64)
            ex, ey = ex / L, ey / L
            fs, fd = src.ravel(), dst.ravel()
            gxm = ((g.gx.ravel()[fs] + g.gx.ravel()[fd]) / 2).reshape(src.shape)
            gym = ((g.gy.ravel()[fs] + g.gy.ravel()[fd]) / 2).reshape(src.shape)
            S.append(fs); D.append(fd); Ls.append(L.ravel())
            DH.append(g.H.ravel()[fd] - g.H.ravel()[fs])
            CR.append(np.abs(-gxm * ey + gym * ex).ravel())
            OK.append(m.ravel())
        self.src, self.dst = np.concatenate(S), np.concatenate(D)
        self.L, self.dh, self.cross = np.concatenate(Ls), np.concatenate(DH), np.concatenate(CR)
        self.move_ok_cells = np.concatenate(OK)
        self.grade = np.abs(self.dh) / self.L
        self.edge_hazard = (self.hazard.ravel()[self.src] + self.hazard.ravel()[self.dst]) / 2

    # helpers
    def node(self, r: int, c: int) -> int:
        return (r - self.g.r0) * self.g.shape[1] + (c - self.g.c0)

    def rc(self, nodes: np.ndarray):
        C = self.g.shape[1]
        return nodes // C + self.g.r0, nodes % C + self.g.c0

    def graph(self, mask: np.ndarray, cost: np.ndarray) -> csr_matrix:
        return csr_matrix((cost[mask], (self.src[mask], self.dst[mask])), shape=(self.n, self.n))

    def accessible_mask(self) -> np.ndarray:
        lim = self.fleet.limits
        return (self.move_ok_cells
                & (self.grade <= math.tan(math.radians(lim.max_grade_deg)))
                & (self.cross <= math.tan(math.radians(lim.max_cross_slope_deg))))


def _window(t: Terrain, a_rc, b_rc, margin_m: float):
    R, C = t.shape
    mr = int(margin_m / t.dy)
    mc = int(margin_m / float(t.dx.mean()))
    r0, r1 = max(min(a_rc[0], b_rc[0]) - mr, 0), min(max(a_rc[0], b_rc[0]) + mr + 1, R)
    c0, c1 = max(min(a_rc[1], b_rc[1]) - mc, 0), min(max(a_rc[1], b_rc[1]) + mc + 1, C)
    return r0, r1, c0, c1


def _walk(pred: np.ndarray, start: int, end: int) -> np.ndarray | None:
    path = [end]
    while path[-1] != start:
        p = pred[path[-1]]
        if p < 0:
            return None
        path.append(p)
    return np.array(path[::-1])


# path reporting

def _moves(ctx: Ctx, nodes: np.ndarray) -> dict:
    """Per-move metrics along a node path."""
    g = ctx.g
    C = g.shape[1]
    r, c = nodes // C, nodes % C
    dr, dc = np.diff(r), np.diff(c)
    dx = g.dx[r[:-1]]
    L = np.hypot(dr * g.dy, dc * dx)
    H = g.H.ravel()
    dh = H[nodes[1:]] - H[nodes[:-1]]
    ex, ey = dc * dx / L, -dr * g.dy / L
    gx, gy = g.gx.ravel(), g.gy.ravel()
    gxm = (gx[nodes[1:]] + gx[nodes[:-1]]) / 2
    gym = (gy[nodes[1:]] + gy[nodes[:-1]]) / 2
    cross = np.abs(-gxm * ey + gym * ex)
    return {"L": L, "dh": dh, "grade": np.abs(dh) / L, "cross": cross}


def _coords(ctx: Ctx, nodes: np.ndarray) -> list[list[float]]:
    r, c = ctx.rc(nodes)
    lon, lat = ctx.t.lonlat(r, c)
    return [[round(float(x), 5), round(float(y), 5)] for x, y in zip(lon, lat)]


def _summary(ctx: Ctx, nodes: np.ndarray) -> dict:
    m = _moves(ctx, nodes)
    lim = ctx.fleet.limits
    hz = ctx.hazard.ravel()[nodes]
    seg_hz = (hz[1:] + hz[:-1]) / 2
    rock = ctx.rock_frac.ravel()[nodes]
    rim = ctx.g.rim_dist.ravel()[nodes]
    H = ctx.g.H.ravel()[nodes]
    length = float(m["L"].sum())
    cum = np.concatenate([[0], np.cumsum(m["L"])])
    step = max(1, len(nodes) // 200)
    return {
        "geometry": {"type": "LineString", "coordinates": _coords(ctx, nodes)},
        "length_km": round(length / 1000, 2),
        "travel_h": round(length / 1000 / ctx.fleet.slowest_kmh, 1),
        "climb_m": round(float(np.clip(m["dh"], 0, None).sum())),
        "max_grade_deg": round(math.degrees(math.atan(float(m["grade"].max(initial=0)))), 1),
        "max_cross_slope_deg": round(math.degrees(math.atan(float(m["cross"].max(initial=0)))), 1),
        "hazard": {
            "max": round(float(hz.max()), 2),
            "mean": round(float((seg_hz * m["L"]).sum() / max(length, 1)), 2),
            "km_above_0_5": round(float(m["L"][seg_hz > 0.5].sum()) / 1000, 2),
            "rock_size_m": lim.max_rock_m,
            "max_rock_cover_pct": round(float(rock.max()) * 100, 2),
            # no mapped crater nearby gives an infinite distance
            "min_crater_rim_distance_m": round(float(rim.min())) if np.isfinite(rim.min()) else None,
        },
        "profile": [[round(float(cum[i]) / 1000, 2), round(float(H[i]))] for i in range(0, len(nodes), step)],
    }


def _snap(ctx: Ctx, node: int) -> tuple[int, float]:
    """Move an endpoint that sits on a blocked cell to the nearest open cell."""
    ok = ctx.cell_ok
    if ok.ravel()[node]:
        return node, 0.0
    if not ok.any():
        return node, 0.0
    _, (ri, ci) = ndimage.distance_transform_edt(~ok, return_indices=True)
    C = ok.shape[1]
    r, c = node // C, node % C
    nr, nc = int(ri[r, c]), int(ci[r, c])
    d = math.hypot((nr - r) * ctx.g.dy, (nc - c) * ctx.g.dx[r])
    return nr * C + nc, d


def _blockers(ctx: Ctx, rr: np.ndarray, cc: np.ndarray) -> dict:
    """Which limits fail on a set of cells (window-local row/col)."""
    lim = ctx.fleet.limits
    out = {}
    if (~ctx.rock_ok[rr, cc]).any():
        share = f"{lim.max_rock_cover_pct}%" if lim.max_rock_cover_pct > 0 else "any measurable share"
        out["rocks"] = f"rocks over {lim.max_rock_m} m cover more than {share} of the ground"
    if (~ctx.rim_ok[rr, cc]).any():
        out["crater_rim"] = f"within {lim.min_crater_rim_clearance_m:.0f} m of a crater rim"
    if (ctx.g.slope[rr, cc] > lim.max_grade_deg).any():
        out["slope"] = f"slope over {lim.max_grade_deg}°"
    if (ctx.g.slope[rr, cc] > lim.max_cross_slope_deg).any() and "slope" not in out:
        out["cross_slope"] = f"side slope over {lim.max_cross_slope_deg}°"
    return out


def _gap(ctx: Ctx, G: csr_matrix, a: int, b: int) -> dict:
    """Closest approach between the areas reachable from each end."""
    _, labels = connected_components(G, directed=False)
    shape = ctx.g.shape
    lab = labels.reshape(shape)
    yy, xx = np.indices(shape)
    pts = lambda m: np.column_stack([yy[m] * ctx.g.dy, xx[m] * float(ctx.g.dx.mean())])

    def edge_cells(n):
        comp = lab == labels[n]
        return comp & ~ndimage.binary_erosion(comp)

    ea, eb = edge_cells(a), edge_cells(b)
    pa, pb = pts(ea), pts(eb)
    d, j = cKDTree(pb).query(pa)
    i = int(np.argmin(d))
    ia = np.flatnonzero(ea.ravel())[i]
    ib = np.flatnonzero(eb.ravel())[int(j[i])]
    C = shape[1]
    n = max(2, int(d[i] / ctx.g.dy) + 2)
    rr = np.linspace(ia // C, ib // C, n).round().astype(int)
    cc = np.linspace(ia % C, ib % C, n).round().astype(int)
    return {
        "geometry": {"type": "LineString", "coordinates": _coords(ctx, np.array([ia, ib]))},
        "width_km": round(float(d[i]) / 1000, 2),
        "blocked_by": _blockers(ctx, rr, cc),
        "reachable_from_start_km2": round(float((lab == labels[a]).sum() * ctx.g.dy * ctx.g.dx.mean()) / 1e6),
    }


# route kinds

def _accessible(ctx: Ctx, a: int, b: int, avoidance: float):
    mask = ctx.accessible_mask()
    G = ctx.graph(mask, ctx.L * (1 + avoidance * ctx.edge_hazard))
    dist_a, pred_a = dijkstra(G, directed=False, indices=a, return_predecessors=True)
    return G, dist_a, pred_a


def _shortest(ctx: Ctx, a: int, b: int, req: RouteRequest, tbm: TBM) -> dict:
    lim, w = ctx.fleet.limits, ctx.fleet.road_width_m
    tg, tc = math.tan(math.radians(lim.max_grade_deg)), math.tan(math.radians(lim.max_cross_slope_deg))
    excess_h = np.clip(np.abs(ctx.dh) - tg * ctx.L, 0, None)
    vol = 0.25 * excess_h * ctx.L * w + (w * w / 8) * np.clip(ctx.cross - tc, 0, None) * ctx.L
    clear = ~ctx.move_ok_cells
    mask = ctx.grade <= math.tan(math.radians(CLIFF_DEG))
    cost = ctx.L * (1 + SHORTEST_AVOIDANCE * ctx.edge_hazard) + req.earthwork_weight * vol + clear * ctx.L * 0.05
    G = ctx.graph(mask, cost)
    dist, pred = dijkstra(G, directed=False, indices=a, return_predecessors=True)
    nodes = _walk(pred, a, b) if np.isfinite(dist[b]) else None
    if nodes is None:
        return {"status": "none", "reason": f"blocked by cliffs steeper than {CLIFF_DEG}°"}

    # Design the road line along the path: never steeper than the fleet's grade, built up with fill wherever the
    # ground is too steep (ramps and embankments), meeting the ground at both ends. Earthwork is the embankment
    # (or, rarely, cut) cross-section along it, plus benching where the side slope is too steep.
    m = _moves(ctx, nodes)
    H = ctx.g.H.ravel()[nodes].astype(float)
    z = _design_profile(H, m["L"], tg)
    d = z - H
    fill_a, cut_a = _section(np.clip(d, 0, None), w), _section(np.clip(-d, 0, None), w)
    fill = (fill_a[1:] + fill_a[:-1]) / 2 * m["L"]
    cut = (cut_a[1:] + cut_a[:-1]) / 2 * m["L"]
    # every metre of road needs a roadbed graded across its full width (deeper where the ground is rough), and a
    # bench cut into any side slope (half cut, half fill), not only where the slope is beyond the fleet's limit
    rough = ctx.g.rough.ravel()[nodes].astype(float)
    level_m = ROADBED_M + np.tan(np.radians((rough[1:] + rough[:-1]) / 2)) * w / 4
    roadbed = w * level_m * m["L"]
    bench = (w * w / 8) * m["cross"] * m["L"]
    v = fill + cut + bench + roadbed
    ok_cells = ctx.cell_ok.ravel()[nodes]
    needs_clear = ~(ok_cells[1:] & ok_cells[:-1])
    rock_bad = ~ctx.rock_ok.ravel()[nodes]
    rock_clear_m = float(m["L"][rock_bad[1:] | rock_bad[:-1]].sum())
    deep = np.abs(d) > 0.5
    work = deep[1:] | deep[:-1] | (bench > 1) | needs_clear
    cum = np.concatenate([[0], np.cumsum(m["L"])])
    segments, i = [], 0
    while i < len(work):
        if not work[i]:
            i += 1
            continue
        j = i
        while j < len(work) and work[j]:
            j += 1
        segments.append({
            "from_km": round(float(cum[i]) / 1000, 2), "to_km": round(float(cum[j]) / 1000, 2),
            "earthwork_m3": round(float(v[i:j].sum())),
            "fill_m3": round(float(fill[i:j].sum())), "cut_m3": round(float(cut[i:j].sum())),
            "max_fill_m": round(float(np.clip(d[i:j + 1], 0, None).max()), 1),
            "max_grade_deg": round(math.degrees(math.atan(float(m["grade"][i:j].max()))), 1),
            "clearing": bool(needs_clear[i:j].any()),
            "coordinates": _coords(ctx, nodes[i: j + 1]),
        })
        i = j
    steep = _steep_choices(ctx, nodes, m, d, fill, cut, bench + roadbed, cum, req, tbm)
    # route totals follow each steep stretch's recommended option; the page re-adds them as the user switches
    earth = float(v.sum())
    length_m = float(m["L"].sum())
    bore_sols = spoil = 0.0
    for st in steep:
        o = st[st["recommended"]]
        earth += o["earthwork_m3"] - st["ramp"]["earthwork_m3"]
        length_m += (o["length_km"] - st["ramp"]["length_km"]) * 1000
        bore_sols += o.get("bore_sols", 0)
        spoil += o.get("spoil_m3", 0)
    step = max(1, len(nodes) // 200)
    out = {"status": "ok", **_summary(ctx, nodes)}
    out.update({
        "length_km": round(length_m / 1000, 2),
        "travel_h": round(length_m / 1000 / ctx.fleet.slowest_kmh, 1),
        "earthwork_m3": round(earth),
        "fill_m3": round(float(fill.sum())), "cut_m3": round(float(cut.sum())), "bench_m3": round(float(bench.sum())),
        "roadbed_m3": round(float(roadbed.sum())), "road_width_m": w,
        "road_area_m2": round(float(m["L"].sum()) * w),
        "max_fill_m": round(float(np.clip(d, 0, None).max()), 1),
        "spoil_fill_m3": round(float(fill.sum())),  # fill that TBM spoil could supply
        "grading_sols": round(earth / req.grading_rate_m3_per_sol, 1),
        "grading_rate_m3_per_sol": req.grading_rate_m3_per_sol,
        "rock_clearing_km": round(rock_clear_m / 1000, 2),
        "clearing_sols": round(rock_clear_m / 1000 / req.rock_clearing_km_per_sol, 1),
        "segments": segments,
        "steep": steep,
        "bore_sols": round(bore_sols, 1),
        "bore_spoil_m3": round(spoil),
        # [km, ground m, road m] along the route, for the profile chart
        "design_profile": [[round(float(cum[k]) / 1000, 2), round(float(H[k])), round(float(z[k]))]
                           for k in list(range(0, len(nodes), step)) + [len(nodes) - 1]],
    })
    return out


ROADBED_M = 0.3      # depth graded and compacted under every metre of built road
REPOSE_DEG = 35.0     # side slopes of embankments and cuts (loose regolith)


def _design_profile(H: np.ndarray, L: np.ndarray, g: float) -> np.ndarray:
    """The lowest road line that never sits below the ground and never exceeds grade g (so steep ground is
    built up with fill), pinned to the ground at both ends (cut only where that pin forces it)."""
    n = len(H)
    s = np.concatenate([[0.0], np.cumsum(L)])
    z = H.copy()
    for i in range(1, n):                       # upper envelope of grade cones from every point
        z[i] = max(z[i], z[i - 1] - g * L[i - 1])
    for i in range(n - 2, -1, -1):
        z[i] = max(z[i], z[i + 1] - g * L[i])
    return np.minimum(z, np.minimum(H[0] + g * s, H[-1] + g * (s[-1] - s)))


def _section(depth: np.ndarray, w: float) -> np.ndarray:
    """Cross-section area (m2) of an embankment or cut of this depth under a road of width w."""
    return w * depth + depth * depth / math.tan(math.radians(REPOSE_DEG))


HAIRPIN = 1.15        # switchback roads run this much longer than their climb needs (turns, landings)
FULL_BENCH_DEG = 30   # steeper side slopes than this won't hold fill, so the road bench is cut in full
STEEP_MERGE = 3       # steep moves this close together are one stretch


def _steep_choices(ctx: Ctx, nodes, m, d, fill, cut, base, cum, req: RouteRequest, tbm: TBM) -> list[dict]:
    """Every stretch where the ground is steeper than the fleet can drive, with three ways past it: a ramp
    of fill along the straight line (the shortest route's design), a switchback road cut into the slope,
    or a straight bore through the hill."""
    lim, w, t = ctx.fleet.limits, ctx.fleet.road_width_m, ctx.t
    tg = math.tan(math.radians(lim.max_grade_deg))
    off = np.abs(d) > 1.0
    steep_mv = np.flatnonzero(off[1:] | off[:-1])
    if not len(steep_mv):
        return []
    runs, s0, prev = [], steep_mv[0], steep_mv[0]
    for k in steep_mv[1:]:
        if k - prev > STEEP_MERGE:
            runs.append((s0, prev + 1)); s0 = k
        prev = k
    runs.append((s0, prev + 1))
    trips = max(ctx.fleet.trips_per_sol, 1)
    drive_cost = lambda km: trips * 2 * km / ctx.fleet.slowest_kmh * HORIZON_SOLS / req.operating_hours_per_sol
    rate, faces = tunnels.bore_rate(tbm), min(req.tbm.count, 2)
    slope = ctx.g.slope.ravel()
    out = []
    for i, j in runs:
        i, j = max(0, i - 1), min(len(m["L"]), j + 1)       # one move of gentler ground each side for portals
        seg = nodes[i: j + 1]
        L = float(m["L"][i:j].sum())
        dh = m["dh"][i:j]
        climb, drop = float(np.clip(dh, 0, None).sum()), float(-np.clip(dh, None, 0).sum())
        side = float(slope[seg].mean())
        f, c, rb = float(fill[i:j].sum()), float(cut[i:j].sum()), float(base[i:j].sum())
        ramp = {"length_km": round(L / 1000, 2), "earthwork_m3": round(f + c + rb), "fill_m3": round(f), "cut_m3": round(c),
                "roadbed_m3": round(rb), "max_fill_m": round(float(np.clip(d[i:j + 1], 0, None).max()), 1),
                "build_sols": round((f + c + rb) / req.grading_rate_m3_per_sol, 1)}

        # road: long enough to gain the height at the fleet's grade, as a bench cut into the slope
        need = (climb + drop) / tg
        hairpins = max(0, math.ceil(need / L) - 1) if need > L else 0
        road_m = max(L, need * (HAIRPIN if hairpins else 1))
        area = w * w * math.tan(math.radians(side)) / (2 if side > FULL_BENCH_DEG else 8)
        road_v = road_m * (area + w * ROADBED_M)
        road = {"length_km": round(road_m / 1000, 2), "hairpins": hairpins, "earthwork_m3": round(road_v),
                "build_sols": round(road_v / req.grading_rate_m3_per_sol, 1), "side_slope_deg": round(side, 1),
                "bench": "full cut" if side > FULL_BENCH_DEG else "half cut, half fill"}

        # bore: straight through a ridge, or up a scarp with a spiral at the top; vehicles drive it,
        # so its grade is held to both the TBM's and the fleet's limit
        (ra, rb), (ca, cb) = ctx.rc(np.array([seg[0], seg[-1]]))
        pa, pb = t.lonlat(ra, ca), t.lonlat(rb, cb)
        pa, pb = (float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1]))
        g_b = min(tbm.max_grade_pct / 100, tg)
        prof = tunnels.transit_profile(t, pa, pb, tbm)
        if not prof["feasible"] or g_b < tbm.max_grade_pct / 100:
            prof = tunnels.climb_profile(t, pa, pb, tbm, g_b)
        bore = {"length_km": round(prof["length_m"] / 1000, 2), "earthwork_m3": 0, "spiral_m": prof.get("spiral_m", 0),
                "grade_pct": round(g_b * 100, 1)}
        if prof["feasible"]:
            sols = prof["length_m"] / (rate * faces)
            bore.update({"status": "ok", "bore_sols": round(sols, 1), "build_sols": round(sols, 1), "faces": faces,
                         "spoil_m3": round(tunnels.face_area(tbm) * prof["length_m"] * tbm.swell),
                         "cover": {k: prof[k] for k in ("min_cover_m", "max_cover_m", "shallow_m", "portal_cut_m")},
                         "path3d": tunnels.path3d(prof, 60), "profile": tunnels.profile(prof, 60)})
        else:
            bore.update({"status": "none", "reason": "the ground between drops below the low end, so the bore would "
                         f"break out {prof['breakout_at_m'] / 1000:.1f} km in"})

        cost = {"ramp": ramp["build_sols"] + drive_cost(ramp["length_km"]),
                "road": road["build_sols"] + drive_cost(road["length_km"])}
        if bore["status"] == "ok":
            cost["bore"] = bore["build_sols"] + drive_cost(bore["length_km"])
        out.append({
            "from_km": round(float(cum[i]) / 1000, 2), "to_km": round(float(cum[j]) / 1000, 2),
            "climb_m": round(climb), "drop_m": round(drop),
            "max_grade_deg": round(math.degrees(math.atan(float(m["grade"][i:j].max()))), 1),
            "coordinates": _coords(ctx, seg),
            "ramp": ramp, "road": road, "bore": bore,
            "recommended": min(cost, key=cost.get),
            "basis": f"build sols + {trips} round trip(s)/sol over {HORIZON_SOLS} sols",
        })
    return out


def _tunnel(ctx: Ctx, a: int, b: int, dist_a: np.ndarray, pred_a, G_acc: csr_matrix,
            req: RouteRequest, tbm: TBM, baseline: float | None, acc_len_m: float | None,
            log: dict | None = None) -> dict:
    """Best route of the form: drive -> portal -> tunnel -> portal -> drive.
    With `log`, records the candidate portals and every portal pair checked."""
    dist_b, pred_b = dijkstra(G_acc, directed=False, indices=b, return_predecessors=True)
    max_m = req.max_tunnel_km * 1000
    spacing = max(2000.0, max_m / 6)
    st = max(1, int(round(spacing / ctx.g.dy)))
    R, C = ctx.g.shape
    sub = np.zeros((R, C), bool)
    sub[::st, ::st] = True
    portal_ok = sub & (ctx.g.slope <= PORTAL_MAX_SLOPE) & ctx.cell_ok
    da, db = dist_a.reshape(R, C), dist_b.reshape(R, C)
    P = np.flatnonzero((portal_ok & np.isfinite(da)).ravel())
    Q = np.flatnonzero((portal_ok & np.isfinite(db)).ravel())
    # the endpoints themselves are always candidate portals
    P = np.union1d(P, [a]) if np.isfinite(dist_a[a]) else P
    Q = np.union1d(Q, [b])
    if log is not None:
        log["portals_from"] = _coords(ctx, _thin(P, TRACE_PORTALS))
        log["portals_to"] = _coords(ctx, _thin(Q, TRACE_PORTALS))
        log["pairs"] = []
    if len(P) == 0 or len(Q) == 0:
        return {"status": "none", "reason": "no portal sites reachable from both ends"}
    dxm = float(ctx.g.dx.mean())
    xy = lambda n: np.column_stack([(n // C) * ctx.g.dy, (n % C) * dxm])
    pairs = cKDTree(xy(P)).sparse_distance_matrix(cKDTree(xy(Q)), max_m, output_type="ndarray")
    pairs = pairs[pairs["v"] >= 500]
    if len(pairs) == 0:
        return {"status": "none", "reason": f"no portal pair within {req.max_tunnel_km} km"}
    pi, qi = P[pairs["i"]], Q[pairs["j"]]
    score = dist_a[pi] + req.tunnel_weight * pairs["v"] + dist_b[qi]
    # pairs are tried best-first even when driving scores better: the best buildable tunnel is still
    # returned, flagged with how it compares to the drive, so it can be weighed instead of hidden
    order = np.argsort(score)[:120]
    t = ctx.t
    tried = skipped_short = 0
    # a straight bore from end to end is tried first, so the tunnel runs A to B when it can, instead of
    # surfacing at a portal partway and driving the rest
    cands = [(int(pi[k]), int(qi[k]), float(score[k])) for k in order]
    (ra, rb), (ca, cb) = ctx.rc(np.array([a, b]))
    d_ab = math.hypot((ra - rb) * ctx.g.dy, (ca - cb) * dxm)
    direct = (f"the straight A–B bore would be {d_ab / 1000:.1f} km, over the {req.max_tunnel_km:g} km limit"
              if d_ab > max_m else None)
    if 500 <= d_ab <= max_m:
        cands.insert(0, (a, b, req.tunnel_weight * d_ab))
    for p, q, sc in cands:
        (pr, qr), (pc, qc) = ctx.rc(np.array([p, q]))
        plon, plat = t.lonlat(pr, pc)
        qlon, qlat = t.lonlat(qr, qc)
        prof = tunnels.transit_profile(t, (float(plon), float(plat)), (float(qlon), float(qlat)), tbm)
        tried += 1
        pair = None
        if log is not None:
            pair = {"from": [round(float(plon), 5), round(float(plat), 5)], "to": [round(float(qlon), 5), round(float(qlat), 5)],
                    "score_km": round(sc / 1000, 2), "result": "breakout"}
            log["pairs"].append(pair)
        if not prof["feasible"]:
            if (p, q) == (a, b):
                direct = (f"the straight A–B bore would come up {prof['breakout_at_m'] / 1000:.1f} km in, where the ground "
                          f"drops faster than the {tbm.max_grade_pct:g}% grade can follow (a crater or valley)")
            continue
        leg_a = _walk(pred_a, a, p)
        leg_b = _walk(pred_b, b, q)
        if leg_a is None or leg_b is None:
            if pair:
                pair["result"] = "unreachable"
            continue
        leg_b = leg_b[::-1]
        faces = min(req.tbm.count, 2)
        rate = tunnels.bore_rate(tbm)
        tun_m = prof["length_m"]
        surf_a = _summary(ctx, leg_a) if len(leg_a) > 1 else None
        surf_b = _summary(ctx, leg_b) if len(leg_b) > 1 else None
        surf_km = (surf_a["length_km"] if surf_a else 0) + (surf_b["length_km"] if surf_b else 0)
        total_m = surf_km * 1000 + tun_m
        if acc_len_m is not None and total_m >= acc_len_m:
            skipped_short += 1
            if pair:
                pair["result"] = "saves too little"
            continue
        if pair:
            pair["result"] = "chosen"
        if acc_len_m is None:
            vs = {"beats_surface": True, "note": "the only way through: no drivable surface route"}
        else:
            saved = (acc_len_m - total_m) / 1000
            better = bool(sc < baseline) and total_m <= MIN_TUNNEL_SAVING * acc_len_m
            vs = {"beats_surface": better, "saves_km": round(saved, 2),
                  "note": f"{saved:.1f} km shorter than driving" if better else
                          f"only {saved:.1f} km shorter than driving; usually not worth the bore"}
        if (p, q) != (a, b) and direct:
            vs["direct"] = direct
        return {
            "status": "ok", **vs,
            "length_km": round(surf_km + tun_m / 1000, 2),
            "travel_h": round((surf_km + tun_m / 1000) / ctx.fleet.slowest_kmh, 1),
            "surface_legs": [s for s in (surf_a, surf_b) if s],
            "tunnel": {
                "geometry": {"type": "LineString", "coordinates": [[round(float(plon), 5), round(float(plat), 5)],
                                                                   [round(float(qlon), 5), round(float(qlat), 5)]]},
                "length_km": round(tun_m / 1000, 2),
                "bore_sols": round(tun_m / (rate * faces), 1),
                "faces": faces,
                "spoil_m3": round(tunnels.face_area(tbm) * tun_m * tbm.swell),
                "cover": {k: prof[k] for k in ("min_cover_m", "max_cover_m", "shallow_m", "portal_cut_m")},
                "profile": tunnels.profile(prof),
            },
        }
    if skipped_short:
        return {"status": "not_needed", "reason": "the drivable route is already about as short as any tunnel line"}
    return {"status": "none", "reason": f"checked {tried} portal pairs; every tunnel would break out to the surface"}


# search trace (for visualizing the calculation)

TRACE_CELLS = 8000      # explored cells returned, thinned evenly through the search order
TRACE_BLOCKED = 6000
TRACE_PORTALS = 400


def _thin(idx: np.ndarray, n: int) -> np.ndarray:
    """Keep at most n entries, evenly spaced, preserving order."""
    if len(idx) <= n:
        return idx
    return idx[np.linspace(0, len(idx) - 1, n).round().astype(int)]


def _trace(ctx: Ctx, a: int, b: int, dist_a: np.ndarray, G_acc: csr_matrix, tunnel_log: dict | None) -> dict:
    """What the accessible search did. Dijkstra settles cells in order of distance, so sorting
    the distance field by value replays its expansion; it stops at the goal, or covers
    everything reachable when the goal can't be reached."""
    R, C = ctx.g.shape
    lim = ctx.fleet.limits
    r0, c0 = ctx.g.r0, ctx.g.c0
    corner = lambda r, c: [round(float(x), 5) for x in ctx.t.lonlat(r, c)]

    blocked = ~ctx.cell_ok | (ctx.g.slope > lim.max_grade_deg)
    reason = np.where(~ctx.rock_ok, 1, np.where(~ctx.rim_ok, 2, 3))   # 1 rocks, 2 crater rim, 3 slope
    bl = _thin(np.flatnonzero(blocked.ravel()), TRACE_BLOCKED)

    reached = np.isfinite(dist_a)
    limit = float(dist_a[b]) if reached[b] else float(dist_a[reached].max())
    seen = np.flatnonzero(reached & (dist_a <= limit))
    seen = _thin(seen[np.argsort(dist_a[seen], kind="stable")], TRACE_CELLS)
    rr, cc = ctx.rc(seen)
    lon, lat = ctx.t.lonlat(rr, cc)
    frac = dist_a[seen] / max(limit, 1.0)

    out = {
        "window": [corner(r0, c0), corner(r0 + R - 1, c0 + C - 1)],
        "grid_cells": R * C,
        "graph_moves": int(len(ctx.src)),
        "drivable_moves": int(G_acc.nnz),
        "blocked_cells": int(blocked.sum()),
        "blocked_by": {"rocks": int((blocked & (reason == 1)).sum()), "crater_rim": int((blocked & (reason == 2)).sum()),
                       "slope": int((blocked & (reason == 3)).sum())},
        "blocked": [[*xy, int(reason.ravel()[n])] for xy, n in zip(_coords(ctx, bl), bl)],
        "settled_cells": int((reached & (dist_a <= limit)).sum()),
        "reached_goal": bool(reached[b]),
        "explored": [[round(float(x), 4), round(float(y), 4), round(float(f), 3)] for x, y, f in zip(lon, lat, frac)],
        "start": _coords(ctx, np.array([a]))[0],
        "goal": _coords(ctx, np.array([b]))[0],
    }
    if tunnel_log is not None:
        out["tunnel"] = tunnel_log
    return out


# entry point

def plan_routes(t: Terrain, req: RouteRequest, vehicles: list[Vehicle], tbm: TBM) -> dict:
    fleet = make_fleet(vehicles, req)
    a_rc, b_rc = t.rc(*req.from_), t.rc(*req.to)
    full = (0, t.shape[0], 0, t.shape[1])
    win = _window(t, a_rc, b_rc, req.search_margin_km * 1000)

    ctx = Ctx(t, t.grid(*win), fleet, req.hazard_weights)
    a, snap_a = _snap(ctx, ctx.node(*a_rc))
    b, snap_b = _snap(ctx, ctx.node(*b_rc))
    G, dist_a, pred_a = _accessible(ctx, a, b, req.avoidance)
    if not np.isfinite(dist_a[b]) and win != full:
        # a detour may leave the search window, so retry on the whole map before giving up
        ctx = Ctx(t, t.grid(*full), fleet, req.hazard_weights)
        a, snap_a = _snap(ctx, ctx.node(*a_rc))
        b, snap_b = _snap(ctx, ctx.node(*b_rc))
        G, dist_a, pred_a = _accessible(ctx, a, b, req.avoidance)
        win = full

    out: dict = {
        "fleet": {
            "limits": fleet.limits.model_dump(),
            "binding_vehicle": fleet.binding,
            "road_width_m": fleet.road_width_m,
            "lanes": fleet.lanes,
            "trips_per_sol": fleet.trips_per_sol,
            "hauler": fleet.hauler.id if fleet.hauler else None,
        },
        "endpoints": {"from_snapped_m": round(snap_a), "to_snapped_m": round(snap_b)},
        "grid_resolution_m": round(t.dy),
        "searched_full_map": win == full,
    }

    baseline = float(dist_a[b]) if np.isfinite(dist_a[b]) else None
    acc_len_m = float(_moves(ctx, _walk(pred_a, a, b))["L"].sum()) if baseline is not None else None
    if "accessible" in req.modes:
        if baseline is not None:
            acc = {"status": "ok", **_summary(ctx, _walk(pred_a, a, b))}
        else:
            acc = {"status": "none", "reason": "no drivable route within the fleet's hazard limits",
                   "gap": _gap(ctx, G, a, b)}
        out["accessible"] = acc
    if "shortest" in req.modes:
        out["shortest"] = _shortest(ctx, a, b, req, tbm)
    tlog = {} if req.trace and "tunnel" in req.modes else None
    if "tunnel" in req.modes:
        out["tunnel"] = _tunnel(ctx, a, b, dist_a, pred_a, G, req, tbm, baseline, acc_len_m, tlog)

    _add_fleet_sizing(out, fleet, req)
    out["recommended"] = _recommend(out, fleet, req, tbm)
    if req.trace:
        out["trace"] = _trace(ctx, a, b, dist_a, G, tlog)
    return out


def _add_fleet_sizing(out: dict, fleet: Fleet, req: RouteRequest):
    if not fleet.trips_per_sol:
        return
    for key in ("accessible", "shortest", "tunnel"):
        r = out.get(key)
        if r and r.get("status") == "ok":
            trip_h = 2 * r["length_km"] / fleet.hauler.speed_kmh
            r["haulers_needed"] = math.ceil(fleet.trips_per_sol * trip_h / req.operating_hours_per_sol)


HORIZON_SOLS = 500


def _recommend(out: dict, fleet: Fleet, req: RouteRequest, tbm: TBM) -> dict | None:
    """Lowest equipment-sols: build time (grading or boring) plus vehicle time spent
    driving the route over the horizon."""
    trips = max(fleet.trips_per_sol, 1)
    cands = []
    for key in ("accessible", "shortest", "tunnel"):
        r = out.get(key)
        if not r or r.get("status") != "ok":
            continue
        build = (r.get("grading_sols", 0) + r.get("clearing_sols", 0) + r.get("tunnel", {}).get("bore_sols", 0)
                 + (r.get("bore_sols", 0) if key == "shortest" else 0))
        drive = trips * 2 * r["travel_h"] * HORIZON_SOLS / req.operating_hours_per_sol
        cands.append((build + drive, key, build, drive))
    if not cands:
        return None
    total, key, build, drive = min(cands)
    return {"route": key, "build_sols": round(build, 1), "driving_vehicle_sols": round(drive, 1),
            "basis": f"build time + {trips} round trip(s)/sol over {HORIZON_SOLS} sols"}
