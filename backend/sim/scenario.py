"""Missions. Nothing is scripted: incidents come from the map's tile odds, the season
and the cargo odds. A seed fixes how the dice fall, so a run can be replayed."""
from __future__ import annotations

from .engine import Sim
from .model import Params


def mission(seed: int | str = 1, **overrides) -> Sim:
    return Sim(Params(**overrides), seed=seed)


random_mission = mission
