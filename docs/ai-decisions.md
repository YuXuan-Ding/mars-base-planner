# AI incident decisions and human override

The construction simulator (`backend/construct`) stops at each incident: a breakdown, boulder field, slope failure,
stuck unit, dust storm, or a TBM cutterhead fault. Someone has to choose a response from a short list of options.
That choice can be made by Claude, by a person, or by a fixed rule.

## Who decides

The Build step's **Incidents decided by** setting picks the mode for a job:

| Mode | Who chooses | How |
|---|---|---|
| **Claude** | Claude (`claude-opus-5-5`, set by `CONSTRUCT_AI_MODEL`) | The page sends each incident to `POST /construct/jobs/{id}/incidents/{iid}/ai-decide`, applies the answer, and shows the choice and reasoning on the incident card. |
| **Me** | The person running the job | The incident card lists the options; the person picks one (`POST .../incidents/{iid}/decide`). **Let Claude decide** asks Claude for that one incident instead. |
| **Fixed rule** | `ai.heuristic` | The option with the smallest expected overrun, then the smallest expected delay plus cost. |

Fast-forward runs (`POST /construct/jobs/{id}/run` with `decide: "ai"` or `"rule"`) decide every incident on the
way without stopping.

## What Claude sees and returns

`construct/ai.py` sends Claude what a site manager would see:
- the job's progress, spend, budget, deadline and projected finish and cost
- the units that are down, and any dust storm
- the incident report and the tile it happened on (slope, rock cover, hazard)
- each option with its estimated delay, cost and risk

Claude must answer with one of the given option ids and one or two sentences on the trade-off (structured output,
`Decision`). An unknown option, a refusal or a failed call falls back to the fixed rule. The decision is then
labelled `rule (AI unavailable)` and carries the error.

## Overriding

In **Me** mode every decision is the person's. In **Claude** mode the person can stop playback, but Claude's choice
is applied as soon as it arrives. Each decision is recorded on the incident with:
- the chosen option
- who decided it: `ai:<model>`, `user`, `rule` or `rule (AI unavailable)`
- the rationale and the outcome

The job's log and its report (`GET /construct/jobs/{id}/report`) list every incident with these fields.

## Where decisions are kept

Jobs, and with them every decision, live in the backend's memory. They are lost when the server restarts, and only
the most recent jobs are kept.
