"""Mars base planner API.

Run:  backend/.venv/bin/uvicorn app.main:app --app-dir backend --port 8001
Docs: http://localhost:8001/docs
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import routing, sites, tunnels
from .catalog import FACILITIES, TBMS, VEHICLES
from .models import TBM, Facility, RouteRequest, SiteRequest, TunnelRequest, Vehicle
from .terrain import get_terrain


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_terrain()  # load and derive layers once, before the first request
    yield


app = FastAPI(title="Mars Base Planner", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from construct.api import router as construct_router  # noqa: E402

app.include_router(construct_router)


def _in_bounds(*pts):
    t = get_terrain()
    for p in pts:
        if not t.contains(*p):
            raise HTTPException(422, f"point {p} is outside the terrain box {t.info()['bounds']}")


def _tbm(model: str) -> TBM:
    if model not in TBMS:
        raise HTTPException(422, f"unknown machine '{model}'; options: {sorted(TBMS)}")
    return TBMS[model]


def _vehicles(items: list[str | Vehicle]) -> list[Vehicle]:
    out = []
    for it in items:
        if isinstance(it, Vehicle):
            out.append(it)
        elif it in VEHICLES:
            out.append(VEHICLES[it])
        else:
            raise HTTPException(422, f"unknown equipment '{it}'; options: {sorted(VEHICLES)}")
    return out


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/terrain/meta")
def terrain_meta():
    return get_terrain().info()


@app.get("/terrain/sample")
def terrain_sample(lon: float = Query(...), lat: float = Query(...)):
    _in_bounds((lon, lat))
    return get_terrain().sample(lon, lat)


@app.get("/terrain/patch")
def terrain_patch(lon: float = Query(...), lat: float = Query(...), half_km: float = Query(20, gt=0, le=40)):
    """Native 200 m elevation around a point (int16 metres, row 0 north, base64), for detail rendering."""
    _in_bounds((lon, lat))
    p = sites.patch(get_terrain(), lon, lat, half_km)
    p["h"] = base64.b64encode(p["h"].tobytes()).decode()
    return p


@app.get("/catalog/facilities")
def catalog_facilities() -> list[Facility]:
    return list(FACILITIES.values())


@app.get("/catalog/machines")
def catalog_machines() -> list[TBM]:
    return list(TBMS.values())


@app.get("/catalog/equipment")
def catalog_equipment() -> list[Vehicle]:
    return list(VEHICLES.values())


@app.post("/routes")
def routes(req: RouteRequest):
    _in_bounds(req.from_, req.to)
    return routing.plan_routes(get_terrain(), req, _vehicles(req.equipment), _tbm(req.tbm.model))


@app.post("/tunnels/plan")
def tunnel_plan(req: TunnelRequest):
    _in_bounds(req.from_, req.to)
    return tunnels.plan(get_terrain(), req, _tbm(req.machine.model))


@app.post("/sites/search")
def site_search(req: SiteRequest):
    if req.polygon:
        # an outline may run off the box edge; the search clips it, but some of it has to be on the map
        if not any(get_terrain().contains(*p) for p in req.polygon):
            raise HTTPException(422, "the outline is outside the terrain box")
    elif req.area:
        _in_bounds(*req.area)
    return sites.search(get_terrain(), req, FACILITIES)
