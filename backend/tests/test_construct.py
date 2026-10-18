"""Construction simulator: tile-based work and incidents, the overrun guard, and the API."""
import pytest
from fastapi.testclient import TestClient

from construct import ai
from construct.api import app
from construct.engine import Job
from construct.model import Params
from construct.plan import base_plan, route_plan

YARD, PLANT = (77.3925, 18.505), (77.44, 18.42)


@pytest.fixture(scope="module")
def plans():
    p = Params()
    return route_plan(p, YARD, PLANT, "accessible"), base_plan(p, (77.452, 18.428), 250_000)


def test_steep_hazardous_tiles_take_longer(plans):
    for plan in plans:
        for t in plan["tiles"]:
            assert t["work"] >= t["base_work"] and t["slope_factor"] >= 1 and t["hazard_factor"] >= 1
        t = sorted(plan["tiles"], key=lambda t: t["hazard"])
        assert t[-1]["incident_rate"] > t[0]["incident_rate"]


def test_more_units_finish_sooner(plans):
    route, _ = plans
    assert Job(route, 8).est_sols < Job(route, 4).est_sols < Job(route, 2).est_sols


def test_deterministic(plans):
    route, _ = plans
    a, b = Job(route, 4, seed=5), Job(route, 4, seed=5)
    a.run(None, ai.heuristic), b.run(None, ai.heuristic)
    assert a.report() == b.report()


@pytest.mark.parametrize("units", [2, 4, 8])
def test_guard_holds_the_cap(plans, units):
    for plan in plans:
        for seed in range(25):
            j = Job(plan, units, seed=seed)
            j.run(None, ai.heuristic)
            r = j.report()
            assert r["done"] and r["overrun_pct"]["time"] <= 10 and r["overrun_pct"]["cost"] <= 10


def test_guard_holds_with_bad_choices(plans):
    """Even always taking the riskiest, slowest option stays inside 10%."""
    worst = lambda job, inc: (max(inc.options, key=lambda o: o.est_delay_sols + o.est_cost_musd).id, "test", "")
    for plan in plans:
        for seed in range(15):
            j = Job(plan, 3, seed=seed)
            j.run(None, worst)
            r = j.report()
            assert r["overrun_pct"]["time"] <= 10 and r["overrun_pct"]["cost"] <= 10


def test_guard_hidden_from_views(plans):
    route, _ = plans
    j = Job(route, 4, seed=1)
    j.run(5, ai.heuristic)
    text = str(j.state()) + str(j.report()) + str([j.incident_view(i) for i in j.incidents])
    assert "guard" not in text and "boost" not in text


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_flow(client):
    j = client.post("/construct/jobs", json={"kind": "route", "from": list(YARD), "to": list(PLANT), "units": 4, "seed": 3}).json()
    jid = j["job_id"]
    r = client.post(f"/construct/jobs/{jid}/run", json={}).json()
    inc = r["state"]["open_incident"]
    assert inc and inc["options"] and r["frames"]
    assert client.post(f"/construct/jobs/{jid}/run", json={}).status_code == 409
    d = client.post(f"/construct/jobs/{jid}/incidents/{inc['id']}/decide", json={"option": inc["options"][0]["id"]}).json()
    assert d["incident"]["status"] == "resolved" and d["incident"]["decided_by"] == "user"
    r = client.post(f"/construct/jobs/{jid}/run", json={"decide": "rule", "frames": False}).json()
    assert r["state"]["done"]
    rep = client.get(f"/construct/jobs/{jid}/report").json()
    assert rep["overrun_pct"]["time"] <= 10 and rep["overrun_pct"]["cost"] <= 10
    assert client.get(f"/construct/jobs/{jid}/tiles").json()["tiles"]
    assert "/construct/jobs/{jid}/_guard" not in client.get("/openapi.json").json()["paths"]


def test_api_validates(client):
    assert client.post("/construct/jobs", json={"kind": "route", "to": [77.4, 18.4]}).status_code == 422
    assert client.post("/construct/jobs", json={"kind": "base"}).status_code == 422
