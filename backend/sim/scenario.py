"""Scenarios: the scripted live demo and random missions for scorecards."""
from __future__ import annotations

from .engine import Sim
from .model import Params

DEMO_SEED = "demo-6"

# The demo story. The planner never sees this script: rollouts treat these events as
# unknown and redraw them from the usual odds.
DEMO_SCRIPT = [
    # Sol ~212: TBM-2 enters a hard band just as its main bearing is going. Symptoms look
    # mostly like hard rock; only an inspection shows the bearing.
    {"kind": "bearing", "sol": 203, "machine": "TBM-2"},
    {"kind": "hard_zone", "machine": "TBM-2", "from_m": 905.0, "to_m": 1010.0},
    {"kind": "note", "sol": 0, "machine": "TBM-2", "notes": ["grinding", "harder_face"]},
    # Sol 300: Earth says the resupply flight (spares) slips 110 sols.
    {"kind": "cargo", "sol": 300, "slip": 110},
    # Sol 384: a strong storm starts during solar conjunction, so Earth can't help.
    {"kind": "storm", "sol": 384, "length": 60, "factor": 0.18},
]


def demo() -> Sim:
    return Sim(Params(random_hard_zones=False), seed=DEMO_SEED, scripted=[dict(e) for e in DEMO_SCRIPT])


def random_mission(seed: int | str, **overrides) -> Sim:
    return Sim(Params(**overrides), seed=seed)


def build(name: str, seed: int | str | None = None, **overrides) -> Sim:
    if name == "demo":
        return demo()
    if name == "random":
        return random_mission(seed if seed is not None else 1, **overrides)
    raise ValueError(f"unknown scenario '{name}' (demo | random)")
