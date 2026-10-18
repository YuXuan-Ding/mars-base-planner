"""Simulator tests: determinism, no hidden-truth leaks, the demo story, and the API."""
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


def test_branches_share_weather():
    """A different decision must not change the dice for later sols."""
    s1 = scenario.demo()
    s1.advance()
    s2 = s1.clone()
    inc = s1.open_incidents[0]
    policy.apply(s1, inc.id, [{"type": "continue"}])
    policy.apply(s2, inc.id, [{"type": "repair", "machine": inc.target, "part": "cutters"}])
    policy.play(s1, policy.fixed_rule)
    policy.play(s2, policy.fixed_rule)
    storms = lambda s: [i.sol for i in s.incidents if i.kind == "dust_storm"]
    assert storms(s1) == storms(s2) == [384]


def test_observe_hides_truth():
    s = scenario.demo()
    s.advance()
    obs = s.observe()
    inc = obs["open_incidents"][0]
    assert inc["kind"] == "tbm_anomaly" and "truth" not in inc
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
    s = scenario.demo()
    s.advance()
    inc = s.open_incidents[0]
    policy.apply(s, inc.id, [{"type": "continue"}])
    assert s.incident_view(inc)["truth"]["cause"] == "bearing"


def test_demo_first_incident_is_ambiguous():
    s = scenario.demo()
    s.advance()
    v = s.incident_view(s.open_incidents[0])
    est = v["cause_estimate"]
    assert 200 <= s.sol <= 220
    assert est["hard_rock"] > 0.3 and est["bearing"] > 0.25      # needs a diagnosis


def test_rollouts_do_not_know_the_script():
    s = scenario.demo()
    s.advance()
    c = policy._rollout_sim(s, 12345)
    assert c.scripted == [] and c.random_kinds == {"storm", "bearing", "cargo"}


def test_demo_fixed_rule_misses_and_guided_makes_it():
    fixed = policy.play(scenario.demo(), policy.fixed_rule)
    assert not fixed["on_time"]
    s = scenario.demo()
    while not s.done:
        for inc in s.advance():
            best = policy.recommend(policy.evaluate(s, policy.menu(s, inc), inc.id, n=120))
            policy.apply(s, inc.id, best["actions"])
            if inc.status == "open":
                s.act(inc.id, [{"type": "continue"}])
    assert s.outcome()["on_time"]


def test_consult_blocked_during_conjunction():
    s = scenario.demo()
    s.sol = 385
    assert not s.comms_ok


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_flow(client):
    r = client.post("/sim/runs", json={"scenario": "demo"}).json()
    rid = r["run_id"]
    r = client.post(f"/sim/runs/{rid}/advance", json={}).json()
    inc = r["state"]["open_incidents"][0]
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
