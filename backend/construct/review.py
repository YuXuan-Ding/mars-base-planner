"""Construction review: Claude writes a report on the hazards a finished job ran into and how
to handle each one better next time.

`facts(job)` collects, for every kind of hazard the job hit (boulders, slope failures,
breakdowns, getting stuck, dust storms): how often, where (the tiles' slope, rock cover and
hazard score), how severe, how each one was handled and by whom, and its estimated cost in
time and money. It also compares incident rates across low-, medium- and high-hazard ground,
which shows whether the terrain hazard map predicted where trouble came. `write(job)` hands
those facts to Claude, which returns the report as Markdown.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from . import ai
from .engine import DT

KIND_NAMES = {"boulder": "Boulder fields", "slope_failure": "Slope failures", "breakdown": "Equipment breakdowns",
              "stuck": "Units stuck in loose ground", "dust_storm": "Dust storms"}
BANDS = (("low", 0.0, 0.3), ("medium", 0.3, 0.6), ("high", 0.6, 1.01))

SYSTEM = """You are the construction lead writing the hazard review of a robotic construction job on
Mars. Your readers plan and run the next job: they want to know which hazards hit, what they cost,
whether the terrain hazard map saw them coming, and the procedures to adopt for each hazard.

Write Markdown with these sections:
# <job> hazard review
A short summary: planned vs actual time and cost, the hazard that cost the most, and the single
most important change.
## Hazards encountered
A table with one row per hazard type: count, where (tile slope / rock cover / hazard ranges),
average severity, how it was handled (choices and who decided), estimated time lost (sols) and
cost ($M).
## Did the hazard map predict it?
Use the hazard-band table: incidents per 10 unit-sols of work on low, medium and high hazard ground,
and the hotspot tiles. Say plainly whether high-hazard ground earned its rating and which terrain
factor (slope, rock cover) lines up with which hazard.
## Procedures by hazard
One subsection per hazard type that occurred. In each:
- Standing response: the default action for this hazard next time, so routine cases need no
  decision, and when to escalate instead.
- Prevention: what to do before or during the job to meet it less often or less severely
  (scouting tiles above a hazard or rock-cover threshold, tile order, equipment, spares, crew size).
- Evidence: incident ids and numbers from this job behind both.
## Open questions
What this single job can't settle.

Use only the facts given; don't invent incidents or numbers. Quote numbers with units (sols, $M, %, deg).
Keep it under about 1,000 words."""


def _span(vals: list[float], nd: int = 1) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if round(lo, nd) == round(hi, nd) else f"{lo:.{nd}f}-{hi:.{nd}f}"


def _storms(job) -> list[dict]:
    """Each storm's start, end and the response chosen."""
    ends = [s for s, t in job.log if t.startswith("Dust storm cleared")]
    out = []
    for inc in (i for i in job.incidents if i.kind == "dust_storm"):
        end = next((e for e in ends if e >= inc.sol), None)
        out.append({"incident": inc.id, "start_sol": inc.sol, "end_sol": end,
                    "length_sols": round(end - inc.sol, 2) if end is not None else None, "response": inc.choice})
    return out


def facts(job) -> dict:
    """Everything the review needs, computed from what the job recorded."""
    rep = job.report()
    status = Counter()
    for f in job.frames:
        for u in f["units"]:
            status[u["status"]] += DT

    by_kind: dict[str, list] = defaultdict(list)
    per_tile: dict[int, list] = defaultdict(list)
    incidents = []
    for inc in job.incidents:
        chosen = next((o for o in inc.options if o.id == inc.choice), None)
        tile = job.tiles[inc.tile] if inc.tile is not None else None
        rec = {
            "id": inc.id, "sol": inc.sol, "kind": inc.kind, "unit": inc.unit, "severity": inc.severity,
            "tile": None if tile is None else {"index": inc.tile, "slope_deg": tile["slope_deg"],
                                               "rock_cover_pct": tile["rock_cover_pct"], "hazard": tile["hazard"]},
            "options": [{"id": o.id, "label": o.label, "est_delay_sols": o.est_delay_sols,
                         "est_cost_musd": o.est_cost_musd, "risk": o.risk or None} for o in inc.options],
            "choice": inc.choice, "decided_by": inc.decided_by, "rationale": inc.rationale, "outcome": inc.outcome,
            "risk_came_true": "Then:" in (inc.outcome or ""),
            "est_delay_sols": chosen.est_delay_sols if chosen else None,
            "est_cost_musd": chosen.est_cost_musd if chosen else None,
        }
        incidents.append(rec)
        by_kind[inc.kind].append(rec)
        if inc.tile is not None:
            per_tile[inc.tile].append(inc.kind)

    hazards = []
    for kind, recs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        tiles = [r["tile"] for r in recs if r["tile"]]
        h = {"hazard": KIND_NAMES.get(kind, kind), "kind": kind, "count": len(recs),
             "avg_severity": round(sum(r["severity"] for r in recs) / len(recs), 2),
             "responses": dict(Counter(r["choice"] for r in recs)),
             "decided_by": dict(Counter(r["decided_by"] for r in recs)),
             "risks_came_true": sum(r["risk_came_true"] for r in recs),
             "est_time_lost_sols": round(sum(r["est_delay_sols"] or 0 for r in recs), 2),
             "est_cost_musd": round(sum(r["est_cost_musd"] or 0 for r in recs), 2),
             "incidents": [r["id"] for r in recs]}
        if tiles:
            h["where"] = {"slope_deg": _span([t["slope_deg"] for t in tiles]),
                          "rock_cover_pct": _span([t["rock_cover_pct"] for t in tiles]),
                          "hazard": _span([t["hazard"] for t in tiles], 2)}
        hazards.append(h)

    bands = []
    for name, lo, hi in BANDS:
        idx = [k for k, t in enumerate(job.tiles) if lo <= t["hazard"] < hi]
        work = sum(job.tiles[k]["work"] for k in idx)
        n = sum(len(per_tile.get(k, [])) for k in idx)
        bands.append({"band": f"{name} ({lo:g}-{min(hi, 1):g})", "tiles": len(idx), "work_unit_sols": round(work, 1),
                      "incidents": n, "per_10_unit_sols": round(10 * n / work, 2) if work else None,
                      "kinds": dict(Counter(kd for k in idx for kd in per_tile.get(k, [])))})

    hot = sorted(per_tile.items(), key=lambda kv: -len(kv[1]))[:5]
    return {
        "job": {"kind": job.plan["kind"], "units": job.n, "tiles": len(job.tiles),
                "length_km": job.plan.get("length_km"), "area_m2": job.plan.get("area_m2"),
                "spare_modules_at_start": job.p.spare_modules, "spare_modules_left": job.spares},
        "plan_vs_actual": {k: rep[k] for k in ("planned", "target", "actual", "overrun_pct")},
        "unit_sols": {k: round(v, 1) for k, v in status.items()},
        "hazards": hazards,
        "hazard_bands": bands,
        "hotspot_tiles": [{"tile": k, **{x: job.tiles[k][x] for x in ("slope_deg", "rock_cover_pct", "hazard")},
                           "incidents": dict(Counter(v))} for k, v in hot],
        "terrain": {"avg_hazard": round(sum(t["hazard"] for t in job.tiles) / len(job.tiles), 3),
                    "max_hazard": max(t["hazard"] for t in job.tiles),
                    "max_slope_deg": max(t["slope_deg"] for t in job.tiles),
                    "max_rock_cover_pct": max(t["rock_cover_pct"] for t in job.tiles)},
        "storms": _storms(job),
        "incidents": incidents,
        "notes": ["Times are in sols; the simulator moves in 0.25-sol steps, so a 0.25-sol gap between an "
                  "incident and its decision is the step size, not waiting time.",
                  "est_delay_sols / est_cost_musd are the planner's estimates for the option chosen; "
                  "outcome says what then happened, including any risk that came true."],
    }


def write(job) -> dict:
    """Returns {"markdown", "model", "hazards"}. Raises RuntimeError if Claude can't be reached."""
    import anthropic
    f = facts(job)
    try:
        resp = ai._get_client().beta.messages.create(
            model=ai.MODEL,
            max_tokens=12000,
            system=SYSTEM,
            messages=[{"role": "user", "content": "Job facts:\n" + json.dumps(f, indent=1)}],
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except (anthropic.AnthropicError, TypeError) as e:
        raise RuntimeError(f"{type(e).__name__}: {str(e)[:200]}") from e
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined to write the review")
    md = "".join(b.text for b in resp.content if b.type == "text").strip()
    return {"markdown": md, "model": resp.model, "hazards": f["hazards"], "hazard_bands": f["hazard_bands"]}
