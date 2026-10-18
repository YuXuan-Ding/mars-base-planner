"""The steep-stretch choices on a made-up mountain: steep flanks with a flat top between them."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from app import routing, tunnels
from app.catalog import TBMS

# these tests weigh bores against roads, so they keep the old slow machine (8 m a sol) the cases were sized for
SLOW_TBM = TBMS["tbm-4m"].model_copy(update={"advance_m_per_sol": 8.0})
from app.models import HazardLimits, RouteRequest


@pytest.fixture(autouse=True)
def steep_bores(monkeypatch):
    # bores on steep stretches are off by default (the bore-through option plans boring); these tests check them
    monkeypatch.setattr(routing, "STEEP_BORES", True)


def _mountain(monkeypatch, flat_km=2.0, **extra):
    step = 200.0
    # 20 moves up at 25%, a flat top, 20 moves down at 25%
    up, flat = 20, int(flat_km * 1000 / step)
    dh = np.array([0.25 * step] * up + [0.0] * flat + [-0.25 * step] * up)
    L = np.full(len(dh), step)
    H = np.concatenate([[0.0], np.cumsum(dh)])
    n = len(H)
    nodes = np.arange(n)
    m = {"L": L, "dh": dh, "grade": np.abs(dh) / L, "cross": np.zeros(len(dh))}
    d = np.zeros(n); d[1:up] = 5; d[up + flat + 1: n - 1] = 5           # off the design line on both flanks only
    zeros = np.zeros(len(dh))
    fill = np.where(np.abs(dh) > 0, 50_000.0, 0.0)                       # ramps up the flanks need a lot of fill
    cum = np.concatenate([[0], np.cumsum(L)])
    lim = HazardLimits(max_grade_deg=12, max_cross_slope_deg=10, max_rock_m=0.5, max_rock_cover_pct=5,
                       min_crater_rim_clearance_m=100)
    fleet = SimpleNamespace(limits=lim, road_width_m=5.0, trips_per_sol=20, slowest_kmh=8.0)   # busy road: bores pay off
    ctx = SimpleNamespace(fleet=fleet, g=SimpleNamespace(slope=np.full(n, 14.0)),
                          rc=lambda ix: (ix, ix), t=SimpleNamespace(lonlat=lambda r, c: (r * 0.002, 0 * r)))
    monkeypatch.setattr(routing, "_coords", lambda ctx, seg: [[float(i), 0.0] for i in seg])

    def prof(length_m, spiral=0.0):
        return {"feasible": True, "length_m": length_m + spiral, "spiral_m": spiral, "z": np.array([0.0, 0.0]), "surf": np.array([0.0, 0.0]),
                "min_cover_m": 20, "max_cover_m": 500, "shallow_m": 0, "portal_cut_m": 0}

    # one bore across the whole mountain is straight; a bore up one flank must spiral up to the top
    def transit(t, a, b, tbm):
        span = abs(b[0] - a[0]) / 0.002 * step
        return prof(span) if span > (up + flat) * step else {**prof(span), "feasible": False, "breakout_at_m": 0}
    monkeypatch.setattr(tunnels, "transit_profile", transit)
    monkeypatch.setattr(tunnels, "climb_profile", lambda t, a, b, tbm, g: prof(abs(b[0] - a[0]) / 0.002 * step, 4000))
    monkeypatch.setattr(tunnels, "path3d", lambda p, k: [])
    monkeypatch.setattr(tunnels, "profile", lambda p, k=0: [])
    req = RouteRequest(**{"from": [0, 0], "to": [1, 0], "equipment": ["hauler"], "tbm": {"model": "tbm-4m", "count": 2}, **extra})
    return routing._steep_choices(ctx, nodes, m, d, fill, zeros, zeros, cum, req, SLOW_TBM)


def test_flanks_with_flat_top_get_one_bore(monkeypatch):
    steep = _mountain(monkeypatch)
    bores = [s for s in steep if s["recommended"] == "bore"]
    assert len(steep) == 1 and bores, steep
    s = bores[0]
    assert s["from_km"] == 0 and s["to_km"] >= 9.5     # foot to foot, under the flat top
    assert s["bore"]["spiral_m"] == 0


def test_far_apart_flanks_stay_separate(monkeypatch):
    # a 30 km plateau is past the tunnel length limit, so each flank is its own stretch
    steep = _mountain(monkeypatch, flat_km=30)
    assert len(steep) == 2


def _flank(monkeypatch, **extra):
    """One side of a mountain: 10 moves at 30% (steeper than the fleet's 12°), then 15 moves at 15% (drivable but
    over half the grade limit), then a flat top."""
    step = 200.0
    dh = np.array([0.30 * step] * 10 + [0.15 * step] * 15 + [0.0] * 10)
    L = np.full(len(dh), step)
    n = len(dh) + 1
    m = {"L": L, "dh": dh, "grade": np.abs(dh) / L, "cross": np.zeros(len(dh))}
    d = np.zeros(n); d[1:10] = 5                                          # only the foot leaves the design line
    fill = np.where(np.arange(len(dh)) < 10, 50_000.0, 0.0)
    base = np.where(dh > 0, 3_000.0, 1_000.0)                             # benching on the sloping ground
    cum = np.concatenate([[0], np.cumsum(L)])
    lim = HazardLimits(max_grade_deg=12, max_cross_slope_deg=10, max_rock_m=0.5, max_rock_cover_pct=5,
                       min_crater_rim_clearance_m=100)
    fleet = SimpleNamespace(limits=lim, road_width_m=5.0, trips_per_sol=20, slowest_kmh=8.0)
    ctx = SimpleNamespace(fleet=fleet, g=SimpleNamespace(slope=np.full(n, 14.0)),
                          rc=lambda ix: (ix, ix), t=SimpleNamespace(lonlat=lambda r, c: (r * 0.002, 0 * r)))
    monkeypatch.setattr(routing, "_coords", lambda ctx, seg: [[float(i), 0.0] for i in seg])
    H = np.concatenate([[0.0], np.cumsum(dh)])

    def climb(t, a, b, tbm, g):
        i, j = int(round(a[0] / 0.002)), int(round(b[0] / 0.002))
        return {"feasible": True, "length_m": abs(j - i) * step, "spiral_m": 0, "z": np.array([H[i], H[j]]), "surf": np.array([H[i], H[j]]),
                "min_cover_m": 20, "max_cover_m": 200, "shallow_m": 0, "portal_cut_m": 0}
    monkeypatch.setattr(tunnels, "transit_profile", lambda t, a, b, tbm: {"feasible": False, "length_m": 0, "breakout_at_m": 0})
    monkeypatch.setattr(tunnels, "climb_profile", climb)
    monkeypatch.setattr(tunnels, "path3d", lambda p, k: [])
    monkeypatch.setattr(tunnels, "profile", lambda p, k=0: [])
    req = RouteRequest(**{"from": [0, 0], "to": [1, 0], "equipment": ["hauler"], "tbm": {"model": "tbm-4m", "count": 2},
                          **extra})
    return routing._steep_choices(ctx, np.arange(n), m, d, fill, np.zeros(len(dh)), base, cum, req, SLOW_TBM)


def test_bore_stays_underground_up_the_sloping_middle(monkeypatch):
    # with climbing inside a tunnel not weighted, a bore up the foot carries on up the sloping middle
    steep = _flank(monkeypatch, tunnel_vertical_m=0)
    assert len(steep) == 1 and steep[0]["recommended"] == "bore", steep
    assert steep[0]["to_km"] == 5.0          # carries on to the flat top, not out at the end of the steep foot (2 km)
    assert steep[0]["bore"]["setup_sols"] > 0


def test_without_the_stay_weight_the_bore_surfaces_early(monkeypatch):
    monkeypatch.setattr(routing, "STAY_WEIGHT", 0.0)
    monkeypatch.setattr(routing, "STAY_UNDER", 10.0)    # no ground counts as sloping: plain runs only
    steep = _flank(monkeypatch, tunnel_vertical_m=0)
    assert steep[0]["to_km"] <= 2.2


def test_climbing_bores_are_avoided(monkeypatch):
    # by default every metre a bore climbs costs 20 m of tunnel: a bore that only climbs the flank loses to a road
    steep = _flank(monkeypatch)
    assert steep[0]["recommended"] != "bore" and steep[0]["bore"]["vertical_m"] > 600


def test_up_and_down_inside_a_bore_count_alike():
    s = np.linspace(0, 1000, 11)
    surf = np.full(11, 100.0)
    down_up = {"z": np.concatenate([np.linspace(100, 50, 6), np.linspace(50, 100, 6)[1:]]), "surf": surf}
    up_only = {"z": np.linspace(100, 200, 11), "surf": np.concatenate([surf[:-1], [200.0]])}
    assert tunnels.vertical_m(down_up) == tunnels.vertical_m(up_only) == 100


def _two_ridges(monkeypatch, valley_moves, **extra):
    """Two steep ridges (5 moves up and 5 down at 30% each) with a flat valley between them; any straight bore
    is feasible and as long as the ground it passes under."""
    step = 200.0
    ridge = [0.30 * step] * 5 + [-0.30 * step] * 5
    dh = np.array(ridge + [0.0] * valley_moves + ridge)
    L = np.full(len(dh), step)
    n = len(dh) + 1
    m = {"L": L, "dh": dh, "grade": np.abs(dh) / L, "cross": np.zeros(len(dh))}
    d = np.where(np.concatenate([[0.0], np.abs(dh)]) > 0, 5.0, 0.0); d[0] = d[-1] = 0
    fill = np.where(np.abs(dh) > 0, 50_000.0, 0.0)
    cum = np.concatenate([[0], np.cumsum(L)])
    lim = HazardLimits(max_grade_deg=12, max_cross_slope_deg=10, max_rock_m=0.5, max_rock_cover_pct=5,
                       min_crater_rim_clearance_m=100)
    fleet = SimpleNamespace(limits=lim, road_width_m=5.0, trips_per_sol=20, slowest_kmh=8.0)
    ctx = SimpleNamespace(fleet=fleet, g=SimpleNamespace(slope=np.full(n, 14.0)),
                          rc=lambda ix: (ix, ix), t=SimpleNamespace(lonlat=lambda r, c: (r * 0.002, 0 * r)))
    monkeypatch.setattr(routing, "_coords", lambda ctx, seg: [[float(i), 0.0] for i in seg])

    def transit(t, a, b, tbm):
        span = abs(b[0] - a[0]) / 0.002 * step
        return {"feasible": True, "length_m": span, "spiral_m": 0, "z": np.array([0.0, 0.0]), "surf": np.array([0.0, 0.0]),
                "min_cover_m": 20, "max_cover_m": 300, "shallow_m": 0, "portal_cut_m": 0}
    monkeypatch.setattr(tunnels, "transit_profile", transit)
    monkeypatch.setattr(tunnels, "path3d", lambda p, k: [])
    monkeypatch.setattr(tunnels, "profile", lambda p, k=0: [])
    req = RouteRequest(**{"from": [0, 0], "to": [1, 0], "equipment": ["hauler"], "tbm": {"model": "tbm-4m", "count": 2},
                          **extra})
    return routing._steep_choices(ctx, np.arange(n), m, d, fill, np.zeros(len(dh)), np.zeros(len(dh)), cum, req,
                                  SLOW_TBM)


def test_resurfacing_penalty_joins_close_bores(monkeypatch):
    with monkeypatch.context() as mp:
        free = _two_ridges(mp, 5, resurface_penalty_sols=0)
    assert [s["recommended"] for s in free] == ["bore", "bore"]           # up for 0.4 km between the ridges
    joined = _two_ridges(monkeypatch, 5)                                    # default penalty
    assert len(joined) == 1 and joined[0]["recommended"] == "bore" and joined[0]["to_km"] == 5.0


def test_resurfacing_penalty_does_not_bore_long_valleys(monkeypatch):
    st = _two_ridges(monkeypatch, 30)
    assert [s["recommended"] for s in st] == ["bore", "bore"]
