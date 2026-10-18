"""Request and catalog schemas."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Point = Annotated[list[float], Field(min_length=2, max_length=2, description="[lon_east, lat]")]


class HazardLimits(BaseModel):
    """What a vehicle can tolerate. A route is refused where any limit is exceeded."""
    max_grade_deg: float = Field(gt=0, le=45, description="steepest climb or descent along the route")
    max_cross_slope_deg: float = Field(gt=0, le=45, description="steepest sideways tilt (tip-over)")
    max_rock_m: float = Field(gt=0, le=5, description="largest rock the vehicle will drive over")
    max_rock_cover_pct: float = Field(ge=0, le=100, description=(
        "ground covered by rocks larger than max_rock_m above which the area is avoided; "
        "0 means avoid any measurable cover"))
    min_crater_rim_clearance_m: float = Field(ge=0, default=0, description="keep at least this far from crater rims")


class HazardOverrides(BaseModel):
    max_grade_deg: float | None = Field(None, gt=0, le=45)
    max_cross_slope_deg: float | None = Field(None, gt=0, le=45)
    max_rock_m: float | None = Field(None, gt=0, le=5)
    max_rock_cover_pct: float | None = Field(None, ge=0, le=100)
    min_crater_rim_clearance_m: float | None = Field(None, ge=0)


class Vehicle(BaseModel):
    id: str
    name: str
    width_m: float = Field(gt=0)
    payload_t: float = Field(ge=0)
    hauls: bool = Field(True, description="counts toward throughput (a crane does not)")
    speed_kmh: float = Field(gt=0)
    limits: HazardLimits


class TBM(BaseModel):
    id: str
    name: str
    diameter_m: float = Field(gt=0)
    advance_m_per_sol: float = Field(gt=0, description="boring advance while running, in average rock")
    duty: float = Field(gt=0, le=1, description="fraction of time boring (power, dust, maintenance)")
    max_grade_pct: float = Field(gt=0)
    min_cover_m: float = Field(ge=0, description="rock cover wanted over the tunnel crown")
    swell: float = Field(ge=1, description="loose spoil volume / solid rock volume")
    power_kw: float = Field(gt=0)


class MachineSelection(BaseModel):
    model: str = "tbm-4m"
    count: int = Field(1, ge=1, le=50)


class HazardWeights(BaseModel):
    slope: float = Field(1.0, ge=0)
    rock: float = Field(1.0, ge=0)
    roughness: float = Field(0.5, ge=0)
    crater: float = Field(0.5, ge=0)


class RouteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: Point = Field(alias="from")
    to: Point
    equipment: list[str | Vehicle] = Field(min_length=1, description="catalog ids or custom vehicles")
    throughput_t_per_sol: float = Field(0, ge=0)
    hazard_overrides: HazardOverrides | None = Field(None, description="fleet-wide limits that replace the vehicles' own")
    hazard_weights: HazardWeights = HazardWeights()
    avoidance: float = Field(3.0, ge=0, description="how far to detour around soft hazards (0 = ignore them)")
    modes: list[Literal["accessible", "shortest", "tunnel"]] = ["accessible", "shortest", "tunnel"]
    tbm: MachineSelection = MachineSelection()
    grading_rate_m3_per_sol: float = Field(400, gt=0, description="surface grading rate (assumed)")
    rock_clearing_km_per_sol: float = Field(0.5, gt=0, description="surface rock clearing rate (assumed)")
    earthwork_weight: float = Field(0.002, ge=0, description="metres of extra length worth avoiding 1 m3 of grading")
    tunnel_weight: float = Field(2.0, ge=1, description="1 m of tunnel costs this many metres of driving")
    max_tunnel_km: float = Field(15, gt=0, le=60)
    operating_hours_per_sol: float = Field(10, gt=0, le=24.6)
    search_margin_km: float = Field(40, gt=0)
    trace: bool = Field(False, description="also return the search steps (window, blocked ground, "
                        "explored cells in search order, tunnel portal pairs tried) for visualizing the calculation")


class Facility(BaseModel):
    """A base facility as ground to flatten: a square pad of `area_m2`."""
    id: str
    name: str
    area_m2: float = Field(gt=0, description="ground to flatten for this facility")
    max_slope_deg: float = Field(gt=0, le=30, description="tilt the finished pad may keep")
    min_separation_m: float = Field(0, ge=0, description="keep every other facility at least this far away (landing pads: plume ejecta)")
    basis: str = Field("", description="where the numbers come from")


class FacilityChoice(BaseModel):
    type: str = Field(description="facility catalog id")
    count: int = Field(1, ge=1, le=12)
    area_m2: float | None = Field(None, gt=0, le=4_000_000, description="override the catalog area")
    max_slope_deg: float | None = Field(None, gt=0, le=30)


class SiteRequest(BaseModel):
    area: list[Point] | None = Field(None, min_length=2, max_length=2,
                                     description="two opposite corners [lon, lat] of a rectangular search area")
    polygon: list[Point] | None = Field(None, min_length=3, max_length=2000,
                                        description="any closed outline [[lon, lat], ...] as the search area instead; "
                                        "only pads that fit wholly inside it count")
    facilities: list[FacilityChoice] = Field(min_length=1, max_length=20)
    max_spread_km: float = Field(3, gt=0, le=20, description="facilities stay within this distance of the base centre "
                                 "(plus their own separation, for landing pads)")
    grading_rate_m3_per_sol: float = Field(400, gt=0, description="surface grading rate (assumed)")
    alternatives: int = Field(3, ge=0, le=10)
    trace: bool = Field(False, description="also return the per-facility cost maps, the candidate centres in "
                        "evaluation order and the layout steps, for visualizing the search")


class TunnelRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    purpose: Literal["mining", "transit", "habitat"]
    from_: Point = Field(alias="from", description="base side; a portal is placed nearby")
    to: Point = Field(description="ore target (mining), far end (transit) or chamber site (habitat)")
    machine: MachineSelection = MachineSelection()
    depth_m: float = Field(30, ge=0, description="mining/habitat: depth of the target below the surface")
    drift_length_m: float = Field(1000, ge=0, description="mining: branch tunnels through the ore body")
    ore_fraction: float = Field(0.3, ge=0, le=1, description="mining: share of drift rock that is ore")
    rock_density_t_m3: float = Field(2.9, gt=0)
    chamber_length_m: float = Field(200, ge=0, description="habitat: length of bored chamber")
    portal_search_radius_km: float = Field(3, gt=0, le=20)
    portal_max_slope_deg: float = Field(5, gt=0, le=30)
