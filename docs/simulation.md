# Construction simulator

A sol-by-sol simulation of building and running a propellant operation, with hidden faults,
so a planner (a person or the AI) has to make decisions under uncertainty. It lives in
`backend/sim/` and has its own API.

```sh
backend/.venv/bin/uvicorn sim.api:app --app-dir backend --port 8002   # docs at /docs
backend/.venv/bin/python -m pytest -q backend/tests/test_sim.py
```

## The mission

**Fuel the return ship before the launch window closes.** Have 1,080 t of propellant in the tanks
and a finished launch pad by **sol 760**. Everything else serves that:

```
solar field ──► power ──────────────────────────────┐
TBM-1, TBM-2 ─► access tunnels ─► ore drifts ─► ore ─► haulers ─► plant ─► propellant
            └─► spoil ─► dozer ─► landing pad                      ▲
crew-hours ─► solar build, plant build, repairs, inspections ─────┘
```

## State

| Part | States and fields | Hidden from the planner |
|---|---|---|
| World | sol, storm (start, sunlight %), Earth contact (lost sols 380–394), resupply ETA, panel dust | storm length, future storms, future cargo slips |
| TBMs (2) | ok / repairing / down / stripped; paused; speed; position; estimated cutter wear; telemetry | true cutter wear, bearing health, hard-rock bands ahead |
| Haulers (3) | ok / repairing / down / stripped; paused; km driven | — |
| Dozer | paused or not | — |
| Plant | building → commissioning → operating ⇄ tripped / restarting; throttle | — |
| Stocks | spares (cutters, bearing, wheel), spoil, ore at portal and plant, propellant, crew-hours owed | — |
| Incidents | open → resolved, with report, symptoms, cause estimate, findings | true cause (revealed on resolve) |

## Incidents

| Incident | Triggered by | What the planner sees |
|---|---|---|
| TBM anomaly | torque, vibration or advance out of range | 3 numbers, estimated wear, a crew log line, and a cause estimate. The real cause is hard rock, worn cutters, or a failing bearing. |
| TBM breakdown | bearing seized or cutters worn out | obvious cause, which spare it needs |
| Wheel damage | per-km odds × the route's rock cover | which hauler, spares left |
| Dust storm | season, or the demo script | sunlight %, duration unknown |
| Plant trip | plant gets under 40 % of its power | needs a 3-sol restart with enough power |
| Cargo slip | Earth notice | new resupply ETA |
| Resupply landed | cargo arrives while machines are down | spares now on hand |

The cause estimate is a proper posterior. It covers combinations, such as a failing bearing
inside a hard band.

## Actions

`continue`, `inspect` (90 % right, 1 sol, 8 crew-h), `consult_earth` (97 % right, 2 sols, not during
conjunction), `inspect_then_fix`, `consult_then_fix`, `repair` (uses a spare), `cannibalize`,
`set_speed`, `pause` / `resume` (optionally until the storm ends), `restart_plant`, `throttle_plant`,
`clean_panels`, `switch_route`. `GET /sim/actions` lists their parameters.

## How odds are computed

`evaluate` replays the rest of the mission many times per option (240 by default, about 3–4 s
for 5 options, spread over the CPU cores). Every replay first **redraws everything hidden**,
consistent with what has been seen:
- the cause of an open anomaly, from its posterior
- true cutter wear and the rock ahead
- the storm length
- whether cargo will slip

Even the demo script is dropped, so the odds never use knowledge the planner couldn't have.
Replay *i* uses the same random draws under every option, so differences between options are
real, not noise. Random draws are keyed by (seed, sol, channel), so rewinding and choosing
differently keeps the same weather.

`recommend` takes the best odds. Options within 2 points count as tied, and ties go to the
option that gathers information, then to the earlier finish.

## Policies and the scorecard

- **fixed_rule:** what a crew without a planner does. It changes cutters at the first alarm,
  repairs whatever breaks, and restarts the plant straight away.
- **sim_guided:** at each incident, scores the menu by replays and takes the recommendation.
  It's a stand-in for the AI with the same information.

Results on 120 random missions (same seeds for both): **fixed rule 52 % on time, sim-guided 84 %**.
The guided policy has about half the TBM alarms, because it doesn't spend scarce spares on the
wrong fix.

## The scripted demo (`scenario: "demo"`)

1. **Sol 211, TBM-2 anomaly.** A failing bearing inside a hard-rock band. The estimate is 59 % hard
   rock / 37 % bearing. Blind fixes score ~60 %. Inspecting or asking Earth scores ~77 % and
   finds the bearing.
2. **Sol 300.** The resupply with spares slips 110 sols.
3. **Sols 307 and 344.** Wheel damage on the rocky shortest route; switching routes is an option.
4. **Sol 384.** A 60-sol storm at 18 % sunlight, while Earth is out of contact. "Keep running" trips
   the plant. Throttling it and pausing the TBMs keeps it alive.

With the recommended choices the mission finishes on **sol 697**. The fixed rule **misses** the
window by about 4 t, mostly from restarting the plant over and over in the storm.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/sim/runs` | new run: `{"scenario": "demo"}` or `{"scenario": "random", "seed": 7, "overrides": {...}}` |
| GET | `/sim/runs/{id}` | observed state |
| POST | `/sim/runs/{id}/advance` | run until an incident, the goal or the deadline |
| GET | `/sim/runs/{id}/incidents/{iid}/options?n=240` | menu with odds and a recommendation |
| POST | `/sim/runs/{id}/evaluate` | score custom options (for the AI) |
| POST | `/sim/runs/{id}/act` | `{"incident_id", "option": i}` or `{"incident_id", "actions": [...]}`; returns findings and, once resolved, the truth |
| GET | `/sim/runs/{id}/forecast` | odds of making it from here under the fixed rule (the gauge) |
| GET | `/sim/runs/{id}/decisions` | decision points so far |
| POST | `/sim/runs/{id}/rewind` | fork a new run from just before an incident |
| GET | `/sim/runs/{id}/series` | per-sol propellant, power share, pad %, solar kW, ore at plant |
| POST | `/sim/scorecard` | many random missions per policy |

Runs live in memory (200 max) and reset when the server restarts.

## Assumptions to keep in mind

All rates are planning assumptions tuned for a good demo, not engineering data. They live in
`backend/sim/model.py` (`Params`). The two places they matter most:
- **Fault odds and symptom models:** bearing onset and decay, how symptoms look for each cause.
- **Power and production balance:** solar size, plant power per tonne, ore per tonne of propellant.
