"""Parameters and state for the construction simulator.

One scenario: fuel the return ship before the launch window closes. The goal is
`target_propellant_t` in the tanks and a finished launch pad by `deadline_sol`.

Every number is a planning assumption, kept here so it can be tuned in one place.
Power is in kW averaged over a sol, work in crew-hours, distances in metres or km.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MASK = (1 << 64) - 1


def u01(*keys: int) -> float:
    """Deterministic uniform [0, 1) from integer keys (splitmix64-style mixing).

    Draws are keyed by (seed, sol, channel, index) rather than taken from a running
    stream, so a branch that makes a different decision still sees the same weather
    and the same dice on later sols. That keeps rewinds and option comparisons fair.
    """
    x = 0x9E3779B97F4A7C15
    for k in keys:
        x = ((x ^ (k & MASK)) * 0xBF58476D1CE4E5B9) & MASK
        x ^= x >> 31
        x = (x * 0x94D049BB133111EB) & MASK
        x ^= x >> 29
    return (x >> 11) / 9007199254740992.0


def gauss(*keys: int) -> float:
    import math
    a = max(u01(*keys, 1), 1e-12)
    b = u01(*keys, 2)
    return math.sqrt(-2 * math.log(a)) * math.cos(2 * math.pi * b)


# random channels
CH_STORM, CH_STORM_LEN, CH_STORM_F, CH_CARGO, CH_BEARING, CH_NOISE, CH_NOTE, CH_WHEEL, \
    CH_FINDING, CH_ZONES, CH_RESAMPLE = range(1, 12)


@dataclass
class Params:
    # goal
    deadline_sol: int = 760
    target_propellant_t: float = 1080.0

    # crew: 4 people x 6 h of outside work per sol
    crew_hours_per_sol: float = 24.0

    # power
    lander_power_kw: float = 300.0        # landed starter array + batteries
    base_load_kw: float = 100.0            # life support, comms; always served first
    solar_target_kw: float = 2000.0
    solar_kw_per_crew_h: float = 1.6       # panel rows deployed per crew-hour
    solar_crew_h_per_sol: float = 12.0
    construction_kw: float = 40.0
    dust_per_sol: float = 0.0004           # panel output lost to settling dust each sol
    dust_max: float = 0.5
    storm_dust: float = 0.08               # extra loss a storm leaves on the panels
    clean_crew_h: float = 10.0

    # TBMs (4 m machine)
    tbm_rate_m: float = 4.8                # m/sol at full speed, duty included
    tbm_power_kw: float = 300.0
    reduced_speed: float = 0.5
    spoil_m3_per_m: float = 17.6           # 12.6 m2 face x 1.4 swell
    ore_t_per_m: float = 8.0               # ore recovered per metre of drift
    cutter_wear_per_m: float = 1 / 900     # one cutter set lasts ~900 m in average rock
    planned_change_wear: float = 0.75     # routine cutter change at this estimated wear, if a spare is on hand
    hard_rate_mult: float = 0.5
    hard_wear_mult: float = 2.0
    bearing_onset_per_sol: float = 0.0012  # chance a main bearing starts to fail, per running sol
    bearing_decay_per_sol: float = 0.05    # health lost per sol once failing (full speed)
    reduced_decay_mult: float = 0.35
    anomaly_cooldown: int = 12
    access_m: dict = field(default_factory=lambda: {"T1": 650.0, "T2": 800.0})
    hard_zones_per_km: float = 0.35
    random_hard_zones: bool = True
    hard_zone_len: tuple = (40.0, 160.0)

    # repairs: (crew-hours, sols out of service)
    cutter_change: tuple = (12.0, 2)
    bearing_replace: tuple = (24.0, 4)
    bearing_seized: tuple = (60.0, 45)
    wheel_repair: tuple = (6.0, 1)
    inspect: tuple = (8.0, 1)
    inspect_accuracy: float = 0.9
    consult_sols: int = 2
    consult_accuracy: float = 0.97

    # haulers
    hauler_payload_t: float = 10.0
    hauler_speed_kmh: float = 4.0
    hauler_load_h: float = 1.0
    hauler_ops_h: float = 10.0
    hauler_power_kw: float = 40.0
    plant_buffer_t: float = 800.0          # haulers stop when this much ore waits at the plant
    ore_margin_t: float = 600.0            # TBMs idle once mined ore covers what is left to make, plus this
    wheel_damage_per_km_pct: float = 0.00005  # per km driven, per % rock cover on the route
    routes: dict = field(default_factory=lambda: {
        "shortest": {"km": 5.1, "rock_cover_pct": 3.5},
        "accessible": {"km": 6.8, "rock_cover_pct": 1.2},
    })
    route: str = "shortest"

    # landing pad
    pad_fill_m3: float = 15000.0
    dozer_m3_per_sol: float = 300.0
    dozer_power_kw: float = 60.0

    # propellant plant
    plant_build_crew_h: float = 1500.0
    plant_crew_h_per_sol: float = 12.0
    commission_sols: int = 15
    plant_rate_t: float = 2.2              # propellant t/sol at full power
    plant_kw_per_t: float = 500.0
    ore_t_per_t: float = 20.0              # 2.5 % water in clay, ~0.5 t water per t propellant
    plant_trip_frac: float = 0.4           # below this share of its power the plant trips offline
    plant_restart: tuple = (10.0, 3)

    # contact with Earth is lost while the Sun is between the planets
    conjunction: tuple = (380, 394)

    # weather
    storm_season: tuple = (250, 560)
    storm_onset_per_sol: float = 0.004
    storm_len: tuple = (20, 60)
    storm_factor: tuple = (0.2, 0.5)       # share of sunlight that still gets through

    # cargo
    resupply_sol: int = 450
    resupply: dict = field(default_factory=lambda: {"cutters": 3, "bearing": 1, "wheel": 4})
    cargo_slip_prob: float = 0.4
    cargo_slip_sols: tuple = (40, 150)
    cargo_notice_sols: tuple = (250, 350)

    spares: dict = field(default_factory=lambda: {"cutters": 4, "bearing": 1, "wheel": 3})
    tbms: tuple = (("TBM-1", "T1"), ("TBM-2", "T2"))
    haulers: tuple = ("H1", "H2", "H3")


@dataclass
class TBM:
    id: str
    tunnel: str
    access_m: float
    wear_per_m: float = 1 / 900        # nominal, what the crew assumes
    position_m: float = 0.0
    m_since_change: float = 0.0
    speed: float = 1.0
    status: str = "ok"                 # ok | repairing | down | stripped
    down_reason: str = ""
    paused: str | None = None          # None | "manual" | "storm_end"
    repair_until: int = -1
    repair_fix: str | None = None
    last_anomaly_sol: int = -999
    ack: dict | None = None            # readings the crew accepted; alarms need worse than this
    telemetry: dict = field(default_factory=dict)
    # hidden truth
    cutter_wear: float = 0.0
    bearing_failing: bool = False
    bearing_health: float = 1.0
    hard_zones: list = field(default_factory=list)

    def in_hard(self) -> bool:
        x = self.position_m
        return any(a <= x < b for a, b in self.hard_zones)

    @property
    def est_wear(self) -> float:
        """Wear the crew would estimate from metres bored, ignoring rock hardness."""
        return self.m_since_change * self.wear_per_m


@dataclass
class Hauler:
    id: str
    status: str = "ok"                 # ok | repairing | down | stripped
    down_reason: str = ""
    paused: str | None = None
    repair_until: int = -1
    repair_fix: str | None = None
    km: float = 0.0


@dataclass
class Plant:
    status: str = "building"           # building | commissioning | operating | tripped | restarting
    build_h: float = 0.0
    commission_left: int = 0
    restart_until: int = -1
    throttle: float = 1.0
    throttle_until: str | None = None  # "storm_end" restores full rate when the storm clears


@dataclass
class Incident:
    id: str
    kind: str                          # tbm_anomaly | tbm_breakdown | wheel_damage | dust_storm | plant_trip | cargo_slip | resupply
    sol: int
    target: str | None
    title: str
    report: str
    symptoms: dict = field(default_factory=dict)
    status: str = "open"               # open | resolved
    actions: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    resolved_sol: int | None = None
    truth: dict = field(default_factory=dict)   # hidden until resolved
