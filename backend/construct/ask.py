"""Ask Claude general questions about the project.

Claude answers from the project's docs (README and docs/*.md) and, when a job id is
given, that job's current state. docs/construction-guard.md is never loaded, so the
assistant knows nothing about the overrun guard.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import ai

ROOT = Path(__file__).resolve().parents[2]
DOCS = ["README.md", "docs/backend.md", "docs/simulation.md", "docs/construction.md"]
MAX_HISTORY = 20

SYSTEM = """You answer questions about a hackathon project: a Mars base planner on real Jezero
crater terrain, with route and tunnel planning, a mission simulator, and a construction
simulator where robotic units build roads and base pads and you (Claude) choose how to
respond to incidents.

Answer from the project documents below and the job state if one is given. Be concise and
specific: numbers, endpoint names, file names. If the documents don't cover something, say
so plainly rather than guessing."""


def _docs() -> str:
    parts = []
    for rel in DOCS:
        p = ROOT / rel
        if p.exists():
            parts.append(f'<document path="{rel}">\n{p.read_text()}\n</document>')
    return "\n\n".join(parts)


def _job_brief(job) -> str:
    st = job.state()
    return json.dumps({
        "plan": {k: v for k, v in job.plan.items() if k not in ("tiles", "geometry")} | {"tiles": len(job.tiles)},
        "state": st,
        "incidents": [job.incident_view(i) for i in job.incidents[-15:]],
    }, indent=1)


def ask(question: str, history: list[dict] | None = None, job=None) -> dict:
    """Returns {"answer", "model"}. Raises RuntimeError if Claude can't be reached."""
    import anthropic
    msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-MAX_HISTORY:]]
    q = question if job is None else f"Current job:\n{_job_brief(job)}\n\nQuestion: {question}"
    msgs.append({"role": "user", "content": q})
    try:
        resp = ai._get_client().beta.messages.create(
            model=ai.MODEL,
            max_tokens=8000,
            system=[{"type": "text", "text": SYSTEM},
                    # the docs rarely change, so cache them across questions
                    {"type": "text", "text": _docs(), "cache_control": {"type": "ephemeral"}}],
            messages=msgs,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except (anthropic.AnthropicError, TypeError) as e:
        raise RuntimeError(f"{type(e).__name__}: {str(e)[:200]}") from e
    if resp.stop_reason == "refusal":
        return {"answer": "Claude declined to answer that question.", "model": resp.model}
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return {"answer": text, "model": resp.model}
