"""HTTP API for the construction simulator.

Run on its own:   backend/.venv/bin/uvicorn construct.api:app --app-dir backend --port 8003
Or mount it:      app.include_router(construct.api.router)

Flow: POST /construct/jobs (a route or a base, and how many units) -> POST .../run to
fast-forward; it stops at each incident, or with `decide: "ai"` lets Claude answer every
incident and keeps going. Answer an incident yourself with .../decide, or ask Claude
with .../ai-decide.
"""
from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from . import ai
from .engine import Job
from .model import Params
from .plan import base_plan, route_plan

router = APIRouter(prefix="/construct", tags=["construction"])

MAX_JOBS = 100
_jobs: "OrderedDict[str, Job]" = OrderedDict()
_ids = itertools.count(1)

Point = tuple[float, float]


class NewJob(BaseModel):
    kind: Literal["route", "base"]
    units: int = Field(4, ge=1, le=40, description="construction units on the job")
    # route
    from_: Point | None = Field(None, alias="from", description="route start [lon, lat]")
    to: Point | None = Field(None, description="route end [lon, lat]")
    mode: Literal["shortest", "accessible"] = "accessible"
    # base
    center: Point | None = Field(None, description="base centre [lon, lat]")
    area_m2: float = Field(250_000, gt=1000, le=25_000_000, description="ground to flatten")
    # targets (default: estimate + 10% contingency)
    budget_musd: float | None = Field(None, gt=0)
    deadline_sols: float | None = Field(None, gt=0)
    seed: int | str = 1
    guard: bool = True

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _where(self):
        if self.kind == "route" and (self.from_ is None or self.to is None):
            raise ValueError("a route needs 'from' and 'to'")
        if self.kind == "base" and self.center is None:
            raise ValueError("a base needs 'center'")
        return self


class Run(BaseModel):
    sols: float | None = Field(None, gt=0, description="fast-forward at most this many sols (default: to the end)")
    decide: Literal["stop", "ai", "rule"] = Field("stop", description="stop at incidents, or let the AI / the rule answer them")
    frames: bool = Field(True, description="return per-step frames (every 0.25 sol) for playback at any speed")


class Decide(BaseModel):
    option: str


def _get(jid: str) -> Job:
    if jid not in _jobs:
        raise HTTPException(404, f"no job '{jid}' (jobs live in memory and reset when the server restarts)")
    _jobs.move_to_end(jid)
    return _jobs[jid]


def _plan_view(job: Job) -> dict:
    pl = {k: v for k, v in job.plan.items() if k != "tiles"}
    pl["tiles"] = len(job.tiles)
    pl["work_unit_sols"] = round(job.work_total, 2)
    return pl


@router.post("/jobs")
def new_job(req: NewJob):
    p = Params()
    try:
        plan = route_plan(p, req.from_, req.to, req.mode) if req.kind == "route" else base_plan(p, req.center, req.area_m2)
    except ValueError as e:
        raise HTTPException(422, str(e))
    job = Job(plan, req.units, p, req.seed, req.budget_musd, req.deadline_sols, guard=req.guard)
    jid = f"job{next(_ids)}"
    _jobs[jid] = job
    while len(_jobs) > MAX_JOBS:
        _jobs.popitem(last=False)
    return {"job_id": jid, "plan": _plan_view(job), "state": job.state()}


@router.get("/jobs/{jid}")
def get_job(jid: str):
    job = _get(jid)
    return {"job_id": jid, "plan": _plan_view(job), "state": job.state()}


@router.get("/jobs/{jid}/tiles")
def tiles(jid: str):
    """Every tile in the job with its slope, hazard, incident odds, work and progress, for the map."""
    job = _get(jid)
    return {"geometry": job.plan["geometry"], "tiles": [
        {**{k: t[k] for k in ("r", "c", "lon", "lat", "slope_deg", "rock_cover_pct", "hazard", "incident_rate", "kinds",
                              "slope_factor", "hazard_factor")},
         "work_unit_sols": t["work"], "left_unit_sols": round(job.remaining[i], 3),
         "done_pct": round(100 * (1 - job.remaining[i] / t["work"]), 1) if t["work"] else 100.0}
        for i, t in enumerate(job.tiles)]}


@router.post("/jobs/{jid}/run")
def run(jid: str, req: Run = Run()):
    """Fast-forward. Returns the frames covered, the incidents decided on the way, and the state."""
    job = _get(jid)
    if job.open_incident and req.decide == "stop":
        raise HTTPException(409, f"answer incident {job.open_incident.id} first, or run with decide 'ai' or 'rule'")
    start = len(job.frames)
    decider = None if req.decide == "stop" else ai.DECIDERS[req.decide]
    handled = job.run(req.sols, decider)
    return {"job_id": jid, "state": job.state(),
            "decided": [job.incident_view(i) for i in handled],
            "frames": job.frames[start:] if req.frames else None,
            "log": [{"sol": s, "text": t} for s, t in job.log[-50:]]}


@router.get("/jobs/{jid}/incidents")
def incidents(jid: str):
    job = _get(jid)
    return [job.incident_view(i) for i in job.incidents]


def _open(job: Job, iid: str):
    inc = next((i for i in job.incidents if i.id == iid), None)
    if inc is None:
        raise HTTPException(404, f"no incident '{iid}'")
    if inc.status != "open":
        raise HTTPException(409, f"incident {iid} is already resolved")
    return inc


@router.post("/jobs/{jid}/incidents/{iid}/ai-decide")
def ai_decide(jid: str, iid: str):
    """Send the incident to Claude, apply its choice, and return its reasoning."""
    job = _get(jid)
    inc = _open(job, iid)
    choice, by, why = ai.claude(job, inc)
    job.decide(iid, choice, by, why)
    return {"incident": job.incident_view(inc), "state": job.state()}


@router.post("/jobs/{jid}/incidents/{iid}/decide")
def decide(jid: str, iid: str, req: Decide):
    """Apply a person's choice (override)."""
    job = _get(jid)
    inc = _open(job, iid)
    try:
        job.decide(iid, req.option, "user")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"incident": job.incident_view(inc), "state": job.state()}


@router.get("/jobs/{jid}/report")
def report(jid: str):
    return _get(jid).report()


@router.get("/jobs/{jid}/_guard", include_in_schema=False)
def guard_log(jid: str):
    """Internal: what the overrun guard did. Not part of the public schema; never sent to the AI."""
    job = _get(jid)
    if not job.guard:
        return {"enabled": False}
    return {"enabled": True, "cap_pct": 100 * job.p.overrun_cap, "boost": job.guard.boost, "log": job.guard.log}


app = FastAPI(title="Mars construction simulator", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/health")
def health():
    return {"ok": True}
