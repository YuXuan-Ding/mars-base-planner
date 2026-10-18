"""Deciders: Claude picks a response to each incident; a rule picks when Claude can't.

Claude sees what a site manager would: the incident report, the tile, the job's progress,
spend and projections against budget and deadline, and the options with their nominal
time and cost effects. It never sees the guard.
"""
from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field

def _load_env():
    """Read KEY=value lines from backend/.env (git-ignored) into the environment."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()
MODEL = os.environ.get("CONSTRUCT_AI_MODEL", "claude-opus-5-5")

SYSTEM = """You are the site manager for robotic construction crews building on Mars.
When an incident is reported you choose one response from the options given.
Weigh time against money: the job has a budget and a deadline, and both matter.
Prefer options that keep the job inside both. Take risk only when the headroom covers it.
Answer with the option id and one or two sentences explaining the trade-off you made."""


class Decision(BaseModel):
    option_id: str = Field(description="id of the chosen option, exactly as given")
    rationale: str = Field(description="one or two sentences on the trade-off")


def _brief(job, inc) -> str:
    st = job.state()
    return json.dumps({
        "job": {"kind": job.plan["kind"], "units": job.n, "sol": st["sol"], "progress_pct": st["progress_pct"],
                "spent_musd": st["spent_musd"], "budget_musd": job.budget, "deadline_sols": job.deadline,
                "projection": st["projection"], "spare_drive_modules": job.spares,
                "units_down": [u["id"] for u in st["units"] if u["status"] == "down"],
                "storm": st["storm"]},
        "incident": job.incident_view(inc),
    }, indent=1)


def heuristic(job, inc) -> tuple[str, str, str]:
    """Smallest expected overrun, then smallest expected delay plus cost."""
    pr = job.projection()
    best, best_s = None, float("inf")
    for o in inc.options:
        t_over = max(0.0, pr["finish_sol"] + o.est_delay_sols - job.deadline) / job.deadline
        c_over = max(0.0, pr["cost_musd"] + o.est_cost_musd - job.budget) / job.budget
        s = 3 * (t_over + c_over) + o.est_delay_sols / job.deadline + o.est_cost_musd / job.budget
        if s < best_s:
            best, best_s = o, s
    return best.id, "rule", f"Lowest expected overrun: about {best.est_delay_sols:.1f} sols and ${best.est_cost_musd:.2f}M."


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def claude(job, inc) -> tuple[str, str, str]:
    """Ask Claude; fall back to the rule if the call fails or returns an unknown option."""
    import anthropic
    ids = [o.id for o in inc.options]
    try:
        resp = _get_client().beta.messages.parse(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Options: {ids}\n\n{_brief(job, inc)}"}],
            output_format=Decision,
            output_config={"effort": "low"},          # one pick per incident; keeps fast-forward quick
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise ValueError(f"no decision (stop_reason {resp.stop_reason})")
        d = resp.parsed_output
        if d.option_id not in ids:
            raise ValueError(f"unknown option '{d.option_id}'")
        return d.option_id, f"ai:{resp.model}", d.rationale
    except (anthropic.AnthropicError, ValueError, TypeError) as e:
        oid, _, why = heuristic(job, inc)
        return oid, "rule (AI unavailable)", f"{why} [AI call failed: {type(e).__name__}: {str(e)[:120]}]"


DECIDERS = {"ai": claude, "rule": heuristic}
