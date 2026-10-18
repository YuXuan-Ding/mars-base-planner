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
    road_m_per_unit_sol: float = 60.0      # road corridor graded, compacted and cleared
    area_m2_per_unit_sol: float = 2500.0   # base ground flattened
    unit_cost_musd_per_sol: float = 0.12   # each unit on the job, working or not
    spare_modules: int = 2                 # swap-in drive modules on hand

    # incidents
    incident_rate_per_unit_sol: float = 0.025  # scaled per tile by hazard and slope
    storm_per_sol: float = 0.008
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
    kind: str                              # breakdown | boulder | slope_failure | stuck | dust_storm
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
