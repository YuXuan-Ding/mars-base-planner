# Design: construction simulator and AI agent

Robotic units build a road or flatten a base pad on the Jezero terrain, tile by tile. Steep, rocky tiles take longer and cause more incidents. When an incident happens, Claude picks the response. Users can also ask Claude general questions about the project.

Code: `backend/construct/`. Tests: `backend/tests/test_construct.py`.

## Goals
- **Simple:** the user picks a job and a number of units, and the sim does the rest.
- **Terrain-driven:** a tile's work time and incident odds come from its own slope and hazard, not from a script.
- **The AI decides:** every incident goes to Claude, which picks an option and explains why. A person can override.
- **Fast-forward:** a whole job runs in about a second without the AI, and it returns frames for playback at any speed.
- **Reproducible:** the same seed and the same decisions give the same run.

Not in scope: real equipment physics, scheduling across several jobs, or keeping jobs after a restart. Jobs live in memory.

## How it fits together

```
 user ──POST /construct/jobs──▶ plan.py ── tiles (work, incident odds) ──▶ engine.Job
                                                                              │ step 0.25 sol
                                                                              ▼
                                 ai.py ◀── incident + options + projection ── incident opens, job pauses
                    (Claude │ rule fallback │ person)
                                   └── choice ──▶ engine.decide() applies the effects ──▶ job resumes

 user ──POST /construct/ask──▶ ask.py ── project docs (+ job state) ──▶ Claude ──▶ answer
```

| File | Role |
|---|---|
| `plan.py` | Turns a road or base pad into tiles, each with its work, incident rate and incident mix |
| `model.py` | Parameters and the shared data types (unit, option, incident) |
| `engine.py` | The job: scheduling, time steps, incidents, applying decisions, views |
| `ai.py` | Incident deciders: Claude, and the rule used as a fallback |
| `ask.py` | Answers general questions about the project |
| `api.py` | HTTP endpoints; runs on its own or mounted in the main app |

## 1. Planning: from terrain to tiles
- **Road:** routed with the main planner (`plan_routes`, using the utility-rover vehicle, `shortest` or `accessible`). The path is sampled every 20 m and split across the 400 m terrain cells it crosses. Base work = metres in the tile ÷ 600 m per unit-sol.
- **Base pad:** a square of the requested area around the centre. Tiles are ordered in a back-and-forth sweep, so a unit's next tile is next to its last. Base work = m² in the tile ÷ 25,000 m² per unit-sol.

Each tile gets a **hazard** score between 0 and 1:

```
hazard = min(1, 0.5·(rock cover % / 5) + 0.3·min(roughness, 1.5) + 0.4·e^(−crater-rim distance / 1500 m))
```

Rock cover is the share of ground covered by rocks 0.25 m and up, using the Golombek–Rapp rock model. Roughness is the cell's roughness divided by the map's 95th percentile.

| Per tile | Formula | Why |
|---|---|---|
| Slope factor | road: `1 + (slope/4)^1.5`; pad: `1 + slope/2` | Flattening a pad means moving more material for each degree of slope |
| Hazard factor | `1 + 2·hazard` | Rocky, rough ground slows digging |
| Work (unit-sols) | `base work × slope factor × hazard factor` | |
| Incident rate (per unit-sol) | `0.25/6 × 0.2 × (0.3 + 2.5·hazard) × (1 + slope/10)` | |
| Incident mix | boulder `0.2 + cover/2`; slope failure `0.05 + (slope/6)²`; breakdown `0.6`; stuck `0.3 + 0.4·max(0, 1 − roughness)` | Smooth ground here is often loose sand |

### Bores
A route job can carry `bores` (`[{coordinates: [portal, …, portal], length_m}]`) and `tbms` (default 2). Each bore
becomes a portal tile at each end (`bore_setup_sols`, 15 TBM-sols) and tiles along its line, each with
`length in the tile ÷ bore_m_per_tbm_sol` (200 m) of work. A TBM takes a free face of a bore (one from each end when
there are two), sets up at its portal, then bores the next tile toward the other end; when the faces meet the bore
breaks through and the unused portal set-up is dropped. TBMs cost `tbm_cost_musd_per_sol` ($0.3M) each, aren't slowed
by storms, and have one incident kind, `tbm_fault` (cutterhead), at `tbm_incident_rate_per_sol` per TBM-sol.

Bores split the road into sides: roads that touch are one side, and a road that touches no portal (a gap the fleet
drives across, a base's internal roads) joins the nearest side. Road units start on the sides in proportion to their
work (at least one each, biggest first) and only take tiles on their own side until a bore between sides breaks
through. The plan estimate runs the same schedule with no incidents or storms (a dry run of the job).

## 2. Running a job
- **Units:**
  - Each unit takes the next tile that hasn't been started. Once none are left, a free unit joins the tile with the most work remaining.
  - Every unit costs $0.12M per sol for the whole job, whether it's working or not.
- **Time:**
  - Steps are 0.25 sol (one 6-hour shift). In each step, every working unit takes `rate × 0.25` unit-sols of work off its tile.
- **Incidents:**
  - For every working unit, the step rolls for an incident at `1 − e^(−rate × exposure)`, using that tile's incident rate.
  - Dust storms hit the whole site, at about 0.00027 per sol (1 in 3,750 sols), and last 3–9 sols.
- **Plan and targets:**
  - The incident-free estimate runs the same schedule with no incidents.
  - By default, the budget and the deadline are that estimate plus 10%. The user can set either one.
- **Fast-forward:**
  - `run` steps until the job finishes, the sol limit is reached, or an incident opens.
  - Each step records a frame with sol, progress, spend, each unit's status and tile, and whether a storm is on. A frontend plays these back at any speed.
- **Determinism:** every random draw is keyed by (seed, step, channel, unit). Choosing a different option never changes the later draws, so runs with different decisions can be compared fairly.

## 3. Incidents and options
When an incident opens, the unit goes down and the job pauses for a decision. Every option shows its estimated delay and cost. Both count units sitting idle, and risky options include their expected cost.

| Incident | Options |
|---|---|
| Breakdown | Repair on site (1.5×sev sols, $0.5M×sev) · Swap a spare drive module (0.25 sol, $0.25M; 2 spares) · Keep working at 60% (25%×sev per sol chance of failing outright) |
| Boulder field | Blast and clear (0.5 sol, $0.3M×sev) · Dig around (+35%×sev of the tile's remaining work) · Realign (roads only: +15%×sev, $0.15M) |
| Slope failure | Shore and regrade · Lay the slope back (+70%×sev work) · Clear and push on (35% per sol chance of a second collapse) |
| Stuck | Tow it out with another unit (both lose time) · Wait for the recovery kit (1.5×sev sols, $0.1M) |
| Dust storm | Pause all units · Work through at half speed (incident odds ×2.5) · Pause and service every unit (ends degraded running, $0.2M) |

Severity (sev) runs from 0.5 to 1.5. When a risky option's risk comes true, the cost and delay are charged at that point, and the result is added to the incident's outcome.

## 4. The AI agent
### Deciding incidents (`ai.py`)
- **What Claude sees:** the same things a site manager would:
  - the incident report and its tile: slope, rock cover, hazard, work left
  - the job's progress, spend, budget, deadline, and projected finish and cost
  - spare parts, units that are down, and the storm status
  - the options with their estimates
- **Call:**
  - It uses `claude-opus-5-5` with structured output `{option_id, rationale}` and low effort, so a decision takes about 6 seconds.
  - If Claude refuses a request, it's automatically retried on another model.
- **Checks:** a refusal, a malformed answer or an option id that isn't on the list all count as a failure.
- **Fallback:** if there's no key or the call fails, the rule picks the option with the lowest expected overrun, then the smallest delay plus cost. The decision is then labelled `rule (AI unavailable)` and gives the error.
- **Ways to decide:**
  - `run` with `decide: "ai"` lets Claude answer every incident as the job runs.
  - `ai-decide` asks Claude about one incident.
  - `decide` applies a person's choice.
  - Every resolved incident records who decided it (`decided_by`) and why.

### Answering questions (`ask.py`)
- **Endpoint:** `POST /construct/ask` with `{question, job_id?, history?}`.
- **Context:**
  - Claude answers from the project docs: `README.md`, `docs/backend.md`, `docs/simulation.md` and this file.
  - Those docs are cached between questions, so follow-ups are cheaper and faster.
  - With a `job_id`, the job's current state and recent incidents come along too, so you can ask things like "why is U3 down?" or "are we on budget?".
- **Follow-ups:** `history` carries the earlier turns of the conversation, up to 20.
- **Settings:** medium effort.
- **Errors:** the endpoint returns 503 if Claude can't be reached. There's no rule fallback for open questions.
- **Setup:** the API key is read from the `ANTHROPIC_API_KEY` environment variable or from `backend/.env`, which git ignores.

## 5. API
Run it on its own with `backend/.venv/bin/uvicorn construct.api:app --app-dir backend --port 8003`. It's also mounted in the main app.

| Endpoint | What it does |
|---|---|
| `POST /construct/jobs` | `{kind: "route", from, to, mode?, units}`, or `{kind: "route", paths, units}` to build along given `[lon, lat]` lines (the web page sends the route the user picked; a tunnel route sends its two road legs), or `{kind: "base", center, area_m2?, units}`; optional `budget_musd`, `deadline_sols`, `seed` |
| `POST /construct/jobs/{id}/run` | Fast-forward: `{sols?, decide: "stop"\|"ai"\|"rule", frames}` |
| `POST /construct/jobs/{id}/incidents/{iid}/ai-decide` | Claude decides this incident |
| `POST /construct/jobs/{id}/incidents/{iid}/decide` | `{option}`: a person decides |
| `POST /construct/ask` | General questions about the project, optionally about one job |
| `GET /construct/jobs/{id}` | Current job state |
| `GET /construct/jobs/{id}/tiles` | Per-tile data for the map |
| `GET /construct/jobs/{id}/incidents` | Every incident so far |
| `GET /construct/jobs/{id}/report` | Planned vs actual time and cost |

## Open questions
- Should the AI weigh options with Monte Carlo rollouts, the way the mission sim does, rather than single estimates?
- Should several jobs share units and spare parts?
- Should running a whole job with the AI use a cheaper model, to make it faster?
