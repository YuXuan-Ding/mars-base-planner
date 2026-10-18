# Construction simulator

`backend/construct/`: units build a road or flatten a base pad tile by tile. Tiles raise incidents, and Claude picks the response.

## How it works
- **Plan** (`plan.py`):
  - A route is planned on the terrain and cut into 400 m tiles. A base is a square of tiles around a centre.
  - Each tile's work is scaled by its slope and its hazard (rock cover, roughness, nearness to a crater rim).
  - The same terrain also sets each tile's incident rate and incident mix: boulders, slope failures, breakdowns and units getting stuck.
- **Run** (`engine.py`):
  - The user picks a number of units. Each one takes the next tile, and idle units help on the biggest tile left.
  - Time moves in 0.25-sol steps. Every unit-sol of work rolls for an incident at that tile's odds. Dust storms hit the whole site.
  - A seed always replays the same run.
- **Decide** (`ai.py`):
  - Each incident comes with 2–3 options, each with its estimated delay and cost.
  - Claude (`claude-opus-5-5`, structured output, server-side fallbacks on) picks one and gives a reason.
  - With no API key, or if the call fails, a rule picks the option with the lowest expected overrun.
- **Targets:** by default the budget and deadline are the incident-free estimate plus 10%.

## Overrun guard (internal)
`guard.py` keeps every job within 10% of its budget and its deadline. It is hidden from the AI's inputs and from the normal responses.
- **How it works:**
  - It lowers incident odds, eases severity, and stops pending risks from firing.
  - It trims charges and speeds crews up, by up to 6×.
  - As a last step, it completes the job before the cap.
- **To check or turn it off:**
  - `GET /construct/jobs/{id}/_guard` returns what it did. That path is not in the OpenAPI schema.
  - `"guard": false` on job creation turns it off.
- **Measured (60 seeds, 2–8 units, rule decider):** with the guard off, the worst overruns were +15% on time and +26% on cost for roads, and +39% on time and +61% on cost for bases. With the guard on, none went over 10%.
- If you present results as the AI's own performance, say the guard is on.

## API
Start it on its own with `backend/.venv/bin/uvicorn construct.api:app --app-dir backend --port 8003`. It is also mounted in the main app (`app.main`).

| Endpoint | What it does |
|---|---|
| `POST /construct/jobs` | `{kind: "route", from, to, mode?, units}` or `{kind: "base", center, area_m2?, units}`. Optional: `budget_musd`, `deadline_sols`, `seed`, `guard`. |
| `POST /construct/jobs/{id}/run` | Fast-forward. Takes `{sols?, decide: "stop"\|"ai"\|"rule", frames}`. The returned frames are 0.25 sol apart, so playback can run at any speed. |
| `POST /construct/jobs/{id}/incidents/{iid}/ai-decide` | Claude decides the incident. |
| `POST /construct/jobs/{id}/incidents/{iid}/decide` | `{option}`: a person decides instead. |
| `GET /construct/jobs/{id}` | Gets the job's current state. |
| `GET /construct/jobs/{id}/tiles` | Per-tile map data. |
| `GET /construct/jobs/{id}/incidents` | Lists the job's incidents. |
| `GET /construct/jobs/{id}/report` | Gets the job report. |
