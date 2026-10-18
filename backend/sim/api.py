"""HTTP API for the construction simulator.

Run on its own:   backend/.venv/bin/uvicorn sim.api:app --app-dir backend --port 8002
Or mount it:      app.include_router(sim.api.router)   (in backend/app/main.py)

A run is an in-memory simulation on the real Jezero map. The planner (a person or the AI)
advances it until an incident opens, looks at the options and their odds, then acts.
Incidents come from the accident odds on the tiles where machines work; nothing is
scripted. Hidden truth is never returned until an incident is resolved.
"""
from __future__ import annotations

import itertools
import statistics
from collections import OrderedDict
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import hazmap, policy, scenario
from .engine import Sim
from .model import Params

router = APIRouter(prefix="/sim", tags=["simulation"])

MAX_RUNS = 200
_runs: "OrderedDict[str, dict]" = OrderedDict()
_ids = itertools.count(1)

ACTIONS = {
    "continue": {"params": {}, "does": "accept the situation and carry on"},
    "inspect": {"params": {}, "does": "TBM anomaly only: stop the machine 1 sol, 8 crew-h; returns a finding (90% right)"},
    "consult_earth": {"params": {}, "does": "TBM anomaly only: Earth engineers review the data, machine waits 2 sols; "
                      "finding 97% right; unavailable during solar conjunction"},
    "inspect_then_fix": {"params": {"machine": "TBM id"}, "does": "inspect, then repair whatever it finds"},
    "consult_then_fix": {"params": {"machine": "TBM id"}, "does": "ask Earth, then repair whatever they find"},
    "repair": {"params": {"machine": "id", "part": "cutters | bearing | wheel"}, "does": "use one spare"},
    "cannibalize": {"params": {"from": "id", "to": "id", "part": "cutters | bearing | wheel"},
                    "does": "strip a machine for the part; the donor is out for good"},
    "set_speed": {"params": {"machine": "TBM id", "speed": "0.5 | 1.0"}, "does": "half speed slows wear and bearing damage"},
    "pause": {"params": {"targets": "[ids or 'dozer']", "until": "storm_end | manual"}, "does": "frees power"},
    "resume": {"params": {"targets": "[ids or 'dozer']"}, "does": "undo a pause"},
    "restart_plant": {"params": {}, "does": "3 sols, 10 crew-h; trips again if power is still under 40%"},
    "throttle_plant": {"params": {"level": "0.2-1.0", "until": "storm_end | manual"}, "does": "plant draws less power"},
    "clean_panels": {"params": {}, "does": "10 crew-h; removes dust from the solar field"},
    "switch_route": {"params": {"route": "shortest | accessible"}, "does": "haul route (km vs rock cover)"},
}


class NewRun(BaseModel):
    seed: int | str = Field(1, description="fixes how the dice fall; same seed and choices give the same run")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Params fields, e.g. "
                                      '{"route": "accessible"} or {"site": {"ore": [lon, lat], "portals": {...}, '
                                      '"plant": [lon, lat], "pad": [lon, lat]}}')


class Advance(BaseModel):
    max_sols: int = Field(10_000, ge=1, le=10_000, description="stop after this many sols even without an incident")


class Act(BaseModel):
    incident_id: str | None = None
    option: int | None = Field(None, ge=0, description="index into the incident's options list")
    actions: list[dict] | None = Field(None, description="custom actions instead of a listed option")


class Option(BaseModel):
    label: str = ""
    actions: list[dict]


class Evaluate(BaseModel):
    incident_id: str | None = None
    options: list[Option] = Field(min_length=1, max_length=12)
    n: int = Field(240, ge=10, le=2000)


class Rewind(BaseModel):
    incident_id: str = Field(description="go back to just before this incident was answered")


class Scorecard(BaseModel):
    seeds: int = Field(48, ge=4, le=400)
    policies: list[Literal["fixed_rule", "sim_guided"]] = ["fixed_rule", "sim_guided"]
    rollouts_per_decision: int = Field(16, ge=4, le=64)
    overrides: dict[str, Any] = Field(default_factory=dict)


# helpers

def _overrides(o: dict) -> dict:
    o = dict(o)
    if "site" in o:   # partial site edits keep the rest of the default layout
        o["site"] = {**Params().site, **o["site"]}
    return o


def _new(sim: Sim, parent: str | None = None) -> str:
    rid = f"run{next(_ids)}"
    _runs[rid] = {"sim": sim, "snapshots": {}, "log_seen": len(sim.log), "parent": parent}
    while len(_runs) > MAX_RUNS:
        _runs.popitem(last=False)
    return rid


def _get(rid: str) -> dict:
    if rid not in _runs:
        raise HTTPException(404, f"no run '{rid}' (runs live in memory and reset when the server restarts)")
    _runs.move_to_end(rid)
    return _runs[rid]


def _incident(sim: Sim, iid: str):
    inc = next((i for i in sim.incidents if i.id == iid), None)
    if inc is None:
        raise HTTPException(404, f"no incident '{iid}'")
    return inc


def _snapshot_open(run: dict):
    sim = run["sim"]
    for inc in sim.open_incidents:
        run["snapshots"].setdefault(inc.id, sim.clone())


def _state(rid: str, run: dict) -> dict:
    sim = run["sim"]
    new = sim.log[run["log_seen"]:]
    run["log_seen"] = len(sim.log)
    return {"run_id": rid, "state": sim.observe(), "new_log": [{"sol": s, "text": t} for s, t in new]}


# endpoints

@router.get("/actions")
def actions():
    return ACTIONS


@router.post("/runs")
def new_run(req: NewRun):
    try:
        sim = scenario.mission(req.seed, **_overrides(req.overrides))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))
    rid = _new(sim)
    return _state(rid, _runs[rid])


@router.get("/runs/{rid}")
def get_run(rid: str):
    run = _get(rid)
    return {"run_id": rid, "state": run["sim"].observe(), "parent": run["parent"]}


@router.post("/runs/{rid}/advance")
def advance(rid: str, req: Advance = Advance()):
    run = _get(rid)
    sim = run["sim"]
    if sim.open_incidents:
        raise HTTPException(409, f"answer open incidents first: {[i.id for i in sim.open_incidents]}")
    sim.advance(req.max_sols)
    _snapshot_open(run)
    return _state(rid, run)


@router.get("/hazard-map")
def hazard_map(lon_west: float = 77.30, lon_east: float = 77.56, lat_south: float = 18.38, lat_north: float = 18.58):
    """Accident odds on every tile of a window (default: around the default site), for drawing."""
    try:
        return hazmap.odds_grid(Params(), lon_west, lat_north, lon_east, lat_south)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/runs/{rid}/map")
def run_map(rid: str):
    """Site layout with per-tile odds, where each machine is, and where incidents happened."""
    sim = _get(rid)["sim"]
    out = hazmap.site_view(sim.site)
    out["route_in_use"] = sim.route
    out["tbm_faces"] = {t.id: {"tunnel": t.tunnel, "position_m": round(t.position_m),
                               "lonlat": [round(x, 5) for x in sim.site.tunnel_point(t.tunnel, t.position_m)]}
                        for t in sim.tbms}
    out["incidents"] = [{"id": i.id, "kind": i.kind, "sol": i.sol, "target": i.target, "status": i.status, **(i.where or {})}
                        for i in sim.incidents if i.where]
    return out


@router.get("/runs/{rid}/series")
def series(rid: str):
    return _get(rid)["sim"].series


@router.get("/runs/{rid}/incidents")
def incidents(rid: str):
    sim = _get(rid)["sim"]
    return [sim.incident_view(i) for i in sim.incidents]


@router.get("/runs/{rid}/incidents/{iid}/options")
def options(rid: str, iid: str, n: int = 240):
    """The ready-made options for an incident, each scored by rollouts, plus a pick."""
    sim = _get(rid)["sim"]
    inc = _incident(sim, iid)
    if inc.status != "open":
        raise HTTPException(409, f"incident {iid} is already resolved")
    opts = policy.menu(sim, inc)
    res = policy.evaluate(sim, opts, iid, n=max(10, min(n, 2000)))
    best = policy.recommend(res)
    return {"incident": sim.incident_view(inc), "options": res, "recommended": res.index(best),
            "note": "Odds come from rollouts that redraw everything hidden from what has been observed."}


@router.post("/runs/{rid}/evaluate")
def evaluate(rid: str, req: Evaluate):
    """Score custom options (e.g. ones the AI composed) the same way."""
    sim = _get(rid)["sim"]
    if req.incident_id:
        inc = _incident(sim, req.incident_id)
        if inc.status != "open":
            raise HTTPException(409, f"incident {req.incident_id} is already resolved")
    res = policy.evaluate(sim, [o.model_dump() for o in req.options], req.incident_id, n=req.n)
    return {"options": res, "recommended": res.index(policy.recommend(res))}


@router.post("/runs/{rid}/act")
def act(rid: str, req: Act):
    run = _get(rid)
    sim = run["sim"]
    if req.incident_id:
        inc = _incident(sim, req.incident_id)
        if inc.status != "open":
            raise HTTPException(409, f"incident {req.incident_id} is already resolved")
    if req.option is not None:
        if not req.incident_id:
            raise HTTPException(422, "option needs incident_id")
        opts = policy.menu(sim, inc)
        if req.option >= len(opts):
            raise HTTPException(422, f"option must be < {len(opts)}")
        acts = opts[req.option]["actions"]
    elif req.actions:
        acts = req.actions
    else:
        raise HTTPException(422, "give option or actions")
    results = policy.apply(sim, req.incident_id, acts)
    out = _state(rid, run)
    out["results"] = results
    if req.incident_id:
        out["incident"] = sim.incident_view(inc)   # includes the revealed truth once resolved
    return out


@router.get("/runs/{rid}/forecast")
def forecast(rid: str, n: int = 240):
    """Chance of making the window from here if the crew follows the fixed rule."""
    return policy.forecast(_get(rid)["sim"], n=max(10, min(n, 2000)))


@router.get("/runs/{rid}/decisions")
def decisions(rid: str):
    run = _get(rid)
    sim = run["sim"]
    return [{"incident_id": iid, "sol": snap.sol, "title": _incident(sim, iid).title,
             "chosen": _incident(sim, iid).actions} for iid, snap in run["snapshots"].items()]


@router.post("/runs/{rid}/rewind")
def rewind(rid: str, req: Rewind):
    """Fork a new run from just before an incident was answered. The original is kept."""
    run = _get(rid)
    snap = run["snapshots"].get(req.incident_id)
    if snap is None:
        raise HTTPException(404, f"no decision point for '{req.incident_id}'")
    sim = snap.clone()
    nid = _new(sim, parent=rid)
    _runs[nid]["log_seen"] = 0
    _runs[nid]["snapshots"] = {k: v.clone() for k, v in run["snapshots"].items()
                               if int(k[1:]) <= int(req.incident_id[1:])}
    return _state(nid, _runs[nid])


def _score_one(args) -> tuple[str, dict]:
    seed, name, n_roll, overrides = args
    sim = scenario.mission(seed, **_overrides(overrides))
    pol = policy.fixed_rule if name == "fixed_rule" else policy.sim_guided(n_roll)
    out = policy.play(sim, pol)
    out["incidents"] = len(sim.incidents)
    return name, out


@router.post("/scorecard")
def scorecard(req: Scorecard):
    """Many random missions per policy. The same seeds are used for every policy."""
    jobs = [(s, name, req.rollouts_per_decision, req.overrides) for name in req.policies for s in range(req.seeds)]
    try:
        results = list(policy._pool().map(_score_one, jobs, chunksize=2))
    except TypeError as e:
        raise HTTPException(422, str(e))
    out = {}
    p = Params(**_overrides(req.overrides))
    for name in req.policies:
        rs = [r for nm, r in results if nm == name]
        fin = sorted(r["finish_sol"] for r in rs if r["finish_sol"] is not None)
        out[name] = {
            "on_time_pct": round(100 * sum(r["on_time"] for r in rs) / len(rs), 1),
            "median_finish_sol": fin[len(fin) // 2] if len(fin) > len(rs) / 2 else None,
            "missed": sum(not r["on_time"] for r in rs),
            "mean_incidents": round(statistics.mean(r["incidents"] for r in rs), 1),
            "mean_shortfall_t_when_missed": round(statistics.mean(
                [p.target_propellant_t - r["propellant_t"] for r in rs if not r["on_time"]] or [0.0]), 1),
        }
    return {"seeds": req.seeds, "deadline_sol": p.deadline_sol, "policies": out,
            "note": "sim_guided scores menu options by rollouts at each incident; it never sees hidden truth."}


app = FastAPI(title="Mars construction simulator", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/health")
def health():
    return {"ok": True}
