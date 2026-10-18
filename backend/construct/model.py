"""Parameters and state for the construction simulator.

Money is in $M, time in sols. Every number is a planning assumption, kept here so it can
be tuned in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MASK = (1 << 64) - 1


def u01(*keys: int) -> float:
    """Deterministic uniform [0, 1) from integer keys (splitmix64-style mixing), so a
    seed replays exactly and a different decision doesn't reshuffle later dice."""
    x = 0x9E3779B97F4A7C15
    for k in keys:
        x = ((x ^ (k & MASK)) * 0xBF58476D1CE4E5B9) & MASK
        x ^= x >> 31
        x = (x * 0x94D049BB133111EB) & MASK
        x ^= x >> 29
    return (x >> 11) / 9007199254740992.0


CH_INCIDENT, CH_KIND, CH_SEV, CH_RISK, CH_STORM, CH_STORM_LEN = range(1, 7)


@dataclass
class Params:
    # productivity of one construction unit (excavator + grader + hauler set) on flat, clean ground
    road_m_per_unit_sol: float = 600.0     # road corridor graded, compacted and cleared
    area_m2_per_unit_sol: float = 25000.0  # base ground flattened
    unit_cost_musd_per_sol: float = 0.12   # each unit on the job, working or not
    # boring machines (TBMs) on routes with bores: a third of a unit's road speed, set up at each portal first
    bore_m_per_tbm_sol: float = 200.0      # main 4 m TBM: 333 m/sol advance x 0.6 duty
    bore_setup_sols: float = 15.0          # launch at a portal (same as the route planner)
    tbm_cost_musd_per_sol: float = 0.3     # each TBM on the job, boring or not
    tbm_incident_rate_per_sol: float = 0.25 / 6 * 0.5 * 0.2   # cutterhead faults per TBM-sol of boring
    spare_modules: int = 2                 # swap-in drive modules on hand

    # incidents
    incident_rate_per_unit_sol: float = 0.25 / 6 * 0.2   # scaled per tile by hazard and slope (per sol of work, so
                                               # incidents per km stay the same at any unit speed)
    storm_per_sol: float = 0.008 / 6 * 0.2   # incidents and storms: 1/6 of the first calibration, then 80% fewer
    storm_len_sols: tuple = (3, 9)

    # plan margins
    contingency: float = 0.10              # default budget and deadline = estimate x (1 + contingency)
    overrun_cap: float = 0.10              # the guard keeps actual <= target x (1 + cap)


@dataclass
class Unit:
    id: str
    status: str = "working"                # working | idle | down | done
    tile: int | None = None
    down_until: float = -1.0
    down_reason: str = ""
    rate: float = 1.0                      # productivity factor on the current tile (degraded running)
    kind: str = "road"                     # road (grading unit) | tbm (boring machine)
    group: int | None = None               # road units: the stretch of road they can reach (bores split it)
    face: tuple | None = None              # TBMs: (bore index, "a" | "b"), the portal they bore from
    risk: dict | None = None               # pending failure risk while running degraded
    worked: float = 0.0                    # unit-sols of work done


@dataclass
class Option:
    id: str
    label: str
    effects: dict                          # what applying it does (see engine._apply)
    est_delay_sols: float = 0.0            # nominal expected effect on finish
    est_cost_musd: float = 0.0             # nominal expected cost, including idle unit time
    risk: str = ""


@dataclass
class Incident:
    id: str
    kind: str                              # breakdown | boulder | slope_failure | stuck | dust_storm | tbm_fault
    sol: float
    unit: str | None
    tile: int | None
    severity: float
    title: str
    report: str
    options: list = field(default_factory=list)
    status: str = "open"                   # open | resolved
    choice: str | None = None
    decided_by: str | None = None
    rationale: str = ""
    outcome: str = ""
