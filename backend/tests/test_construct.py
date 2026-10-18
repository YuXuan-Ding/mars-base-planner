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


def test_ask_validates(client):
    assert client.post("/construct/ask", json={"question": ""}).status_code == 422
    assert client.post("/construct/ask", json={"question": "hi", "job_id": "nope"}).status_code == 404


def test_ask_context_never_mentions_guard():
    from construct import ask
    assert "guard" not in ask._docs().lower()


def test_job_follows_given_paths(client):
    legs = [[[77.78, 18.30], [77.80, 18.31], [77.83, 18.33]], [[77.90, 18.36], [77.93, 18.38]]]
    r = client.post("/construct/jobs", json={"kind": "route", "mode": "tunnel", "paths": legs, "units": 3})
    assert r.status_code == 200, r.text
    tiles = client.get(f"/construct/jobs/{r.json()['job_id']}/tiles").json()
    assert tiles["geometry"]["type"] == "MultiLineString"
    # every tile lies near the given lines, none in the gap between the legs (the bored stretch)
    assert all(t["lon"] < 77.84 or t["lon"] > 77.89 for t in tiles["tiles"])
    assert client.post("/construct/jobs", json={"kind": "route", "paths": [[[77.8, 18.3]]]}).status_code == 422


def test_review_facts(plans):
    from construct import review
    route, _ = plans
    j = Job(route, 4, seed=7)
    j.run(None, ai.heuristic)
    f = review.facts(j)
    assert f["incidents"] and f["unit_sols"]["working"] > 0 and f["plan_vs_actual"]["actual"]["sols"] == j.finish_sol
    assert "guard" not in str(f).lower() and "boost" not in str(f)
    kinds = {h["kind"] for h in f["hazards"]}
    assert kinds == {i.kind for i in j.incidents}
    assert sum(h["count"] for h in f["hazards"]) == len(j.incidents)
    assert sum(b["tiles"] for b in f["hazard_bands"]) == len(j.tiles)


def test_review_endpoints(client):
    jid = client.post("/construct/jobs", json={"kind": "route", "from": list(YARD), "to": list(PLANT), "units": 4}).json()["job_id"]
    assert client.post(f"/construct/jobs/{jid}/review").status_code == 409      # not finished
    assert client.get(f"/construct/jobs/{jid}/review").status_code == 404       # none written yet


def test_review_pdf(client):
    from construct.api import _jobs
    from construct.pdf import render
    assert render("# T\n\n**A**\n- x ≥ 1\n\n| a | b |\n|---|---|\n| 1 | 2 |").startswith(b"%PDF")
    jid = client.post("/construct/jobs", json={"kind": "route", "from": list(YARD), "to": list(PLANT), "units": 4}).json()["job_id"]
    _jobs[jid].review = {"markdown": "# Job hazard review\n\nAll fine.", "model": "test"}
    r = client.get(f"/construct/jobs/{jid}/review", params={"format": "pdf"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    assert "pdf" not in client.get(f"/construct/jobs/{jid}/review").json()


def _bored_route():
    """Road from YARD to a portal, a 6 km bore, road on to beyond the far portal."""
    p = Params()
    P, Q = (77.40, 18.49), (77.455, 18.47)
    paths = [[YARD, P], [Q, (77.47, 18.44)]]
    return p, route_plan(p, None, None, "through", paths, [{"coordinates": [P, Q]}])


def test_tbms_bore_from_both_portals():
    p, plan = _bored_route()
    b = plan["bores"][0]
    assert plan["groups"] == 2 and b["groups"] == [0, 1]
    j = Job(plan, 4, p, seed=2, n_tbms=2)
    assert {u.face for u in j.tbms} == {(0, "a"), (0, "b")}
    assert {u.group for u in j.road_units} == {0, 1}          # crews start on both sides
    # set-up at both portals in parallel, then the bore from both ends
    assert j.est_sols >= p.bore_setup_sols + b["length_m"] / (2 * p.bore_m_per_tbm_sol) - 0.5
    one = Job(plan, 4, p, seed=2, n_tbms=1)
    assert one.est_sols > j.est_sols


def test_road_crews_wait_for_the_breakthrough():
    p, plan = _bored_route()
    j = Job(plan, 4, p, seed=4, n_tbms=2)
    j.run(None, ai.heuristic)
    broke = next(s for s, t in j.log if "broke through" in t)
    group = {i: t.get("group") for i, t in enumerate(j.tiles)}
    first = {}
    for f in j.frames:
        if f["sol"] >= broke:
            break
        for u in f["units"]:
            if u["kind"] == "road" and u["tile"] is not None:
                first.setdefault(u["id"], group[u["tile"]])
                assert group[u["tile"]] == first[u["id"]], f"{u['id']} crossed the bore before it broke through"
    assert j.done


def test_storms_do_not_stop_boring():
    p, plan = _bored_route()
    j = Job(plan, 2, p, seed=1, n_tbms=2)
    j.storm = {"start": 0, "end": 99, "mode": "pause", "factor": 0.5}
    for u in j.road_units:
        u.status, u.down_reason = "down", "sheltering"
    before = j.bore_left()
    for _ in range(80):
        j.step()
    assert j.bore_left() < before and j.road_left() == pytest.approx(sum(j.tiles[i]["work"] for i in j.road_idx))
