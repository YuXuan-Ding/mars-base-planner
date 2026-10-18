"""API tests on the real Jezero terrain. Inside the crater floor -> outside the east rim."""
import base64
import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

INSIDE, INSIDE_2, OUTSIDE = [77.45, 18.45], [77.52, 18.47], [78.2, 18.4]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def route(client, **body):
    r = client.post("/routes", json={"from": INSIDE, "to": OUTSIDE, **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_catalogs(client):
    eq = client.get("/catalog/equipment").json()
    assert {"hauler", "crane", "crew-rover"} <= {v["id"] for v in eq}
    assert all("max_rock_m" in v["limits"] for v in eq)
    assert client.get("/catalog/machines").json()[0]["diameter_m"] > 0


def test_out_of_bounds(client):
    r = client.post("/routes", json={"from": [10, 10], "to": OUTSIDE, "equipment": ["hauler"]})
    assert r.status_code == 422


def test_fleet_takes_strictest_limits(client):
    j = route(client, equipment=["crew-rover", "crane"], modes=["accessible"])
    lim, who = j["fleet"]["limits"], j["fleet"]["binding_vehicle"]
    assert lim["max_rock_m"] == 0.15 and who["max_rock_m"] == "crane"
    assert lim["max_grade_deg"] == 8 and who["max_grade_deg"] == "crane"


def test_rover_has_accessible_route(client):
    j = route(client, equipment=["crew-rover"])
    acc = j["accessible"]
    assert acc["status"] == "ok"
    assert acc["max_grade_deg"] <= 18 and acc["max_cross_slope_deg"] <= 12
    assert acc["hazard"]["max_rock_cover_pct"] <= 3.0
    assert acc["geometry"]["type"] == "LineString"
    assert j["shortest"]["length_km"] <= acc["length_km"]


def test_rock_size_limit_blocks_and_explains(client):
    j = route(client, equipment=["crew-rover"], modes=["accessible", "shortest"],
              hazard_overrides={"max_rock_m": 0.2, "max_rock_cover_pct": 0})
    assert j["fleet"]["binding_vehicle"]["max_rock_m"] == "override"
    assert j["accessible"]["status"] == "none"
    assert "rocks" in j["accessible"]["gap"]["blocked_by"]
    assert j["shortest"]["rock_clearing_km"] > 0


def test_stricter_rock_limit_never_gets_rockier(client):
    loose = route(client, equipment=["utility-rover"], modes=["accessible"])["accessible"]
    strict = route(client, equipment=["utility-rover"], modes=["accessible"],
                   hazard_overrides={"max_rock_m": 0.3, "max_rock_cover_pct": 2.5})["accessible"]
    if strict["status"] == "ok":
        assert strict["hazard"]["max_rock_cover_pct"] <= 2.5
        assert strict["length_km"] >= loose["length_km"] - 0.5


def test_shortest_reports_grading(client):
    j = route(client, equipment=["utility-rover"], modes=["shortest"], hazard_overrides={"max_grade_deg": 2.5})
    s = j["shortest"]
    assert s["earthwork_m3"] > 0 and s["grading_sols"] > 0 and s["segments"]


def test_tunnel_when_surface_blocked(client):
    j = route(client, equipment=["tbm-carrier"])
    assert j["accessible"]["status"] == "none"
    t = j["tunnel"]
    assert t["status"] in ("ok", "none")
    if t["status"] == "ok":
        assert t["tunnel"]["bore_sols"] > 0 and t["tunnel"]["spoil_m3"] > 0


def test_throughput_sizes_fleet(client):
    j = route(client, equipment=["hauler", "crew-rover"], throughput_t_per_sol=300, modes=["shortest"])
    assert j["fleet"]["hauler"] == "hauler" and j["fleet"]["trips_per_sol"] == 30
    assert j["fleet"]["lanes"] == 2
    assert j["shortest"]["haulers_needed"] >= 1


@pytest.mark.parametrize("purpose,extra", [
    ("mining", {"depth_m": 40, "drift_length_m": 1500}),
    ("habitat", {"depth_m": 20, "chamber_length_m": 300}),
])
def test_tunnel_plan(client, purpose, extra):
    j = client.post("/tunnels/plan", json={"purpose": purpose, "from": INSIDE, "to": INSIDE_2,
                                            "machine": {"model": "tbm-4m", "count": 3}, **extra}).json()
    assert j["status"] == "ok", j
    assert j["sols"] > 0 and j["spoil_m3"] > 0 and len(j["path3d"]) > 2
    if purpose == "mining":
        assert j["ore_t"] > 0


def test_more_machines_finish_mining_sooner(client):
    body = {"purpose": "mining", "from": INSIDE, "to": INSIDE_2, "drift_length_m": 2000}
    one = client.post("/tunnels/plan", json={**body, "machine": {"model": "tbm-4m", "count": 1}}).json()
    four = client.post("/tunnels/plan", json={**body, "machine": {"model": "tbm-4m", "count": 4}}).json()
    assert four["sols"] < one["sols"]
    assert four["spoil_m3"] == one["spoil_m3"]


def test_trace_replays_the_search(client):
    j = route(client, equipment=["hauler"], trace=True)
    tr = j["trace"]
    assert tr["settled_cells"] > 0 and 0 < len(tr["explored"]) <= 8000
    fr = [p[2] for p in tr["explored"]]
    assert fr == sorted(fr) and fr[-1] <= 1.0          # cells come back in search order
    assert tr["blocked_cells"] == sum(tr["blocked_by"].values())
    assert all(p[2] in (1, 2, 3) for p in tr["blocked"])
    assert len(tr["window"]) == 2 and tr["drivable_moves"] <= tr["graph_moves"]
    if j["tunnel"]["status"] == "ok":
        assert any(p["result"] == "chosen" for p in tr["tunnel"]["pairs"])
    assert "trace" not in route(client, equipment=["hauler"], modes=["accessible"])


SITE_AREA = [[77.3, 18.2], [77.75, 18.6]]


def test_site_search_finds_flat_ground(client):
    body = {"area": SITE_AREA, "facilities": [{"type": "habitat"}, {"type": "landing-pad"}, {"type": "solar-farm"}],
            "trace": True}
    j = client.post("/sites/search", json=body).json()
    assert j["status"] == "ok", j
    best, pads = j["best"], j["best"]["facilities"]
    assert len(pads) == 3 and best["earthwork_m3"] == sum(p["earthwork_m3"] for p in pads)
    assert all(a["earthwork_m3"] >= best["earthwork_m3"] for a in j["alternatives"])
    lon0, lon1 = sorted(x[0] for x in SITE_AREA)
    assert all(lon0 <= p["center"][0] <= lon1 for p in pads)
    pad = next(p for p in pads if p["facility"] == "landing-pad")
    for p in pads:
        if p is not pad:   # landing pad keeps 1 km from everything
            d = math.hypot((p["center"][0] - pad["center"][0]) * 3.21e4, (p["center"][1] - pad["center"][1]) * 5.93e4 / 1.0)
            assert d >= 900
    tr = j["trace"]
    assert {m["facility"] for m in tr["cost_maps"]} == {"habitat", "landing-pad", "solar-farm"}
    assert len(tr["layout_steps"]) == 3 and tr["centres"]
    ranked = [c for c in tr["centres"] if c[3] == 0]
    assert len(ranked) == 1 and ranked[0][2] == best["earthwork_m3"]


def test_flatter_tolerance_never_costs_more(client):
    one = lambda s: client.post("/sites/search", json={"area": SITE_AREA, "alternatives": 0,
                                "facilities": [{"type": "habitat", "max_slope_deg": s}]}).json()["best"]["earthwork_m3"]
    assert one(10) <= one(1)


def test_site_search_rejects_unknown_facility(client):
    r = client.post("/sites/search", json={"area": SITE_AREA, "facilities": [{"type": "castle"}]})
    assert r.status_code == 422


def test_terrain_patch(client):
    p = client.get("/terrain/patch", params={"lon": 77.5, "lat": 18.4, "half_km": 10}).json()
    assert p["rows"] > 50 and p["cols"] > 50 and p["resolution_m"] == 200
    assert len(base64.b64decode(p["h"])) == p["rows"] * p["cols"] * 2


def test_route_without_nearby_craters_does_not_crash(client):
    # no mapped crater near the route used to make the rim distance infinite and fail the request
    r = client.post("/routes", json={"from": [77.4632, 16.7999], "to": [77.6922, 16.6712], "equipment": ["hauler"]})
    assert r.status_code == 200


def test_steep_stretches_offer_road_or_bore(client):
    # the ends are ~1,400 m apart in height, so some climb steeper than the fleet can drive is unavoidable
    j = client.post("/routes", json={"from": [76.08, 21.006], "to": [76.36, 21.006], "equipment": ["hauler"]}).json()
    steep = j["shortest"]["steep"]
    assert steep
    for st in steep:
        assert st["road"]["length_km"] >= st["ramp"]["length_km"] - 0.01 and st["road"]["earthwork_m3"] > 0
        assert st["ramp"]["fill_m3"] + st["ramp"]["cut_m3"] > 0
        assert st["recommended"] in ("road", "bore")
        if st["bore"]["status"] == "ok":
            assert st["bore"]["bore_sols"] > 0 and st["bore"]["path3d"]


def test_tunnel_is_offered_even_when_driving_wins(client):
    j = client.post("/routes", json={"from": INSIDE, "to": INSIDE_2, "equipment": ["crew-rover"]}).json()
    t = j["tunnel"]
    assert t["status"] in ("ok", "not_needed", "none")
    if t["status"] == "ok":
        assert "beats_surface" in t and t["note"]


def test_site_search_inside_drawn_outline(client):
    import math
    cx, cy = 77.45, 18.45
    poly = [[cx + 0.12 * math.cos(a), cy + 0.12 * math.sin(a)] for a in (i / 20 * 2 * math.pi for i in range(20))]
    j = client.post("/sites/search", json={"polygon": poly, "facilities": [{"type": "landing-pad", "count": 1},
                                                                          {"type": "habitat", "count": 1}]}).json()
    assert j["status"] == "ok"
    for pad in j["best"]["facilities"]:
        for lo, la in pad["polygon"]:
            assert math.hypot(lo - cx, la - cy) <= 0.12 + 1e-3
    assert client.post("/sites/search", json={"facilities": [{"type": "habitat", "count": 1}]}).status_code == 422


def test_shortest_goes_around_a_high_steep_peak(client):
    # straight across a ~2,400 m peak; going round it climbs much less for a little more length
    j = route(client, **{"from": [77.20, 16.734], "to": [77.44, 16.734]}, equipment=["hauler"], modes=["shortest"])
    s = j["shortest"]
    assert s["status"] == "ok"
    assert s["climb_m"] < 800                       # straight over climbs ~1,200 m
    assert max(p[1] for p in s["design_profile"]) < -1500   # the summit is at about -800 m


PEAK = {"from": [77.20, 16.734], "to": [77.44, 16.734], "equipment": ["hauler"]}


def test_minimal_route_builds_least(client):
    j = route(client, **PEAK, modes=["shortest", "minimal"])
    s, m = j["shortest"], j["minimal"]
    assert m["status"] == "ok"
    assert m["earthwork_m3"] < s["earthwork_m3"] and m["grading_sols"] <= s["grading_sols"]


def test_bore_through_the_peak_pays_off(client):
    # boring is a third of a grading unit's speed, so a level bore under the peak beats the road around it
    for tp in (0, 300):
        j = route(client, **PEAK, throughput_t_per_sol=tp, modes=["shortest", "through"])
        t = j["through"]
        assert t["status"] == "ok" and t["beats_shortest"] and j["recommended"]["route"] == "through"
        assert t["tunnel_km"] > 5 and all(b["vertical_m"] < 200 for b in t["bores"])   # level, under the peak


def test_through_costs_a_given_height(client):
    none = client.post("/routes/through", json={**PEAK, "no_bore": True}).json()
    assert none["status"] == "ok" and none["bores"] == [] and none["height_m"] is None
    line = none["line"]
    span = line["highest_ground_m"] - line["lowest_bore_m"]
    hi = client.post("/routes/through", json={**PEAK, "height_m": line["lowest_bore_m"] + 0.7 * span}).json()
    lo = client.post("/routes/through", json={**PEAK, "height_m": line["lowest_bore_m"] + 0.2 * span}).json()
    assert hi["status"] == lo["status"] == "ok"
    assert lo["tunnel_km"] > hi["tunnel_km"] > 0 and lo["climb_m"] < hi["climb_m"] < none["climb_m"]
    assert all(b["height_m"] == lo["height_m"] for b in lo["bores"])
    # the route never changes: roads and bores together cover the same straight line
    for p in (none, hi, lo):
        on_line = sum(l["design_profile"][-1][0] for l in p["legs"]) + p["tunnel_km"]
        assert on_line == pytest.approx(line["length_km"], abs=0.05)
    bad = client.post("/routes/through", json={**PEAK, "height_m": line["lowest_bore_m"] - 100}).json()
    assert bad["status"] == "none" and "below the ground" in bad["reason"]


def test_through_falls_back_to_the_rule_without_claude(client, monkeypatch):
    from construct import ai

    def broken():
        raise TypeError("no client in tests")
    monkeypatch.setattr(ai, "_get_client", broken)
    j = client.post("/routes/through", json={**PEAK, "decide": "ai"}).json()
    assert j["decided_by"] == "rule (AI unavailable)" and "AI call failed" in j["rationale"]
