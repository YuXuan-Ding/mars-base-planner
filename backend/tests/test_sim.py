"""Simulator tests: determinism, no hidden-truth leaks, map-based incidents, and the API."""
import pytest
from fastapi.testclient import TestClient

from sim import policy, scenario
from sim.api import app
from sim.engine import Sim


def run_fixed(seed):
    s = scenario.random_mission(seed)
    return policy.play(s, policy.fixed_rule), s


def test_deterministic():
    a, sa = run_fixed(7)
    b, sb = run_fixed(7)
    assert a == b and sa.series["propellant_t"] == sb.series["propellant_t"]


def _first_incident(seed=1):
    s = scenario.mission(seed)
    while not s.open_incidents:
        s.advance()
    return s


def test_branches_share_weather():
    """A different decision must not change the dice for later sols."""
    s1 = scenario.mission(2)
    s2 = s1.clone()
    policy.play(s1, policy.fixed_rule)
    policy.play(s2, lambda sim, inc: [{"type": "continue"}])
    storms = lambda s: [i.sol for i in s.incidents if i.kind == "dust_storm"]
    assert storms(s1)[:1] == storms(s2)[:1]


def test_observe_hides_truth():
    s = _first_incident()
    obs = s.observe()
    assert obs["open_incidents"] and "truth" not in obs["open_incidents"][0]
    hidden = {"bearing_health", "bearing_failing", "hard_zones", "cutter_wear", "truth"}

    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield k
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)
    assert hidden.isdisjoint(set(keys(obs)))


def test_truth_revealed_after_resolution():
    s = _first_incident()
    inc = s.open_incidents[0]
    policy.apply(s, inc.id, [{"type": "continue"}])
    assert "truth" in s.incident_view(inc)


def test_incidents_come_from_map_tiles():
    s = scenario.mission(1)
    policy.play(s, policy.fixed_rule)
    tiled = [i for i in s.incidents if i.kind in ("tbm_anomaly", "wheel_damage", "tbm_breakdown")]
    assert tiled and all(i.where and "tile_odds" in i.where for i in tiled)
    hard = [s.site.tiles[ti]["hard_per_km"] for t in s.site.tunnels.values() for _, _, ti in t["segs"]]
    assert max(hard) > 1.5 * min(hard)          # the map varies, so the odds vary


def test_rollouts_redraw_hidden_state():
    s = _first_incident()
    c = policy._rollout_sim(s, 12345)
    assert c.seed == 12345 and c.site is s.site


def test_guided_beats_fixed_rule():
    fixed = guided = 0
    for seed in range(6):
        fixed += policy.play(scenario.mission(seed), policy.fixed_rule)["on_time"]
        guided += policy.play(scenario.mission(seed), policy.sim_guided(8))["on_time"]
    assert guided >= fixed


def test_consult_blocked_during_conjunction():
    s = scenario.mission(1)
    s.sol = 385
    assert not s.comms_ok


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_flow(client):
    r = client.post("/sim/runs", json={"seed": 1}).json()
    rid = r["run_id"]
    r = client.post(f"/sim/runs/{rid}/advance", json={}).json()
    inc = r["state"]["open_incidents"][0]
    assert client.get(f"/sim/runs/{rid}/map").json()["tunnels"]
    assert client.get("/sim/hazard-map").json()["layers"]["wheel_per_km"]
    assert client.post(f"/sim/runs/{rid}/advance", json={}).status_code == 409
    opts = client.get(f"/sim/runs/{rid}/incidents/{inc['id']}/options", params={"n": 40}).json()
    assert len(opts["options"]) >= 4 and 0 <= opts["recommended"] < len(opts["options"])
    done = client.post(f"/sim/runs/{rid}/act", json={"incident_id": inc["id"], "option": opts["recommended"]}).json()
    assert done["incident"]["status"] == "resolved" and "truth" in done["incident"]
    fork = client.post(f"/sim/runs/{rid}/rewind", json={"incident_id": inc["id"]}).json()
    assert fork["state"]["sol"] == r["state"]["sol"] and fork["state"]["open_incidents"][0]["id"] == inc["id"]
    custom = client.post(f"/sim/runs/{fork['run_id']}/evaluate", json={
        "incident_id": inc["id"], "n": 20,
        "options": [{"label": "slow", "actions": [{"type": "set_speed", "machine": inc["target"], "speed": 0.5}]}]}).json()
    assert 0 <= custom["options"][0]["p_on_time"] <= 1
    assert 0 <= client.get(f"/sim/runs/{rid}/forecast", params={"n": 20}).json()["p_on_time"] <= 1


def test_api_scorecard(client):
    r = client.post("/sim/scorecard", json={"seeds": 8, "rollouts_per_decision": 4}).json()
    assert set(r["policies"]) == {"fixed_rule", "sim_guided"}
