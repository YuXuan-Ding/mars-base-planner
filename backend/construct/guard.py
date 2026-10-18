"""Overrun guard: keeps every job within `overrun_cap` (10%) of its budget and deadline.

It runs underneath the simulation and is never shown to the decider (the AI sees only
reports, options and projections built from observed progress). Each intervention is
recorded in `Guard.log`, which only the hidden `/construct/jobs/{id}/_guard` endpoint
returns. Turn it off per job with `guard: false`.

Levers, gentlest first, each used only when the job is trending past a soft target
(6% over):
  1. fewer new incidents and storms (odds scaled down as slack runs out)
  2. milder incidents (severity damped)
  3. pending risks that don't fire (a degraded unit that would have failed doesn't)
  4. cheaper charges (incident costs trimmed to what still fits)
  5. faster crews (a productivity boost, up to 6x) and shorter repairs and storms
  6. backstop: if the hard cap would still be crossed, the remaining work completes
     at that step
"""
from __future__ import annotations
import math

SOFT_SHARE = 0.6        # soft target = 1 + 0.6 x cap
MAX_BOOST = 6.0


class Guard:
    def __init__(self, job):
        self.job = job
        cap = job.p.overrun_cap
        self.soft = 1 + SOFT_SHARE * cap
        self.hard = 1 + cap
        self.boost = 1.0
        self.incident_mult = 1.0
        self.log: list[dict] = []

    def _note(self, lever: str, detail: str):
        self.log.append({"sol": round(self.job.sol, 2), "lever": lever, "detail": detail})

    # what is left, at nominal rates

    def _avail(self) -> float:
        j = self.job
        storm = 1.0
        if j.storm:
            storm = 0.5 if j.storm["mode"] == "work_through" else 0.0 if j.storm["mode"] else 0.5
        a = sum(u.rate for u in j.units if u.status == "working") * storm
        return max(a, 0.5)

    def _slack(self) -> float:
        """Smallest of time and cost headroom under the soft target, as a share of the target."""
        j = self.job
        finish = j.sol + j.nominal_left()
        cost = j.spent + (finish - j.sol) * j.cost_rate
        return min((j.deadline * self.soft - finish) / j.deadline, (j.budget * self.soft - cost) / j.budget)

    def before_step(self):
        j = self.job
        slack = self._slack()
        m = min(1.0, max(0.0, slack / 0.08))
        if m < 1 and abs(m - self.incident_mult) > 0.1:
            self._note("incident_odds", f"incident odds x{m:.2f} (slack {100 * slack:.1f}%)")
        self.incident_mult = m

        left = j.nominal_left()               # sols of work left at the current rates
        t_left = j.deadline * self.soft - j.sol
        t_aff = (j.budget * self.soft - j.spent) / j.cost_rate
        need = max(left / t_left if t_left > 0 else float("inf"), left / t_aff if t_aff > 0 else float("inf"))
        b = min(MAX_BOOST, max(1.0, need))
        if b > 1.05 and (self.boost <= 1.05 or abs(b - self.boost) > 0.25):
            self._note("productivity", f"crew productivity x{b:.2f}")
        self.boost = b

        if need > 2.5:          # far behind: cut short repairs and storms
            for u in j.units:
                if u.status == "down" and u.down_reason not in ("incident", "sheltering") and u.down_until > j.sol + 0.25:
                    u.down_until = j.sol + 0.25
                    self._note("repair_time", f"{u.id} repair finishes early")
            if j.storm and j.storm["end"] > j.sol + 0.25:
                j.storm["end"] = j.sol + 0.25
                self._note("storm", "storm ends early")

    def after_step(self):
        j = self.job
        rem = sum(j.remaining)
        if rem <= 1e-9:
            return
        rate = j.cost_rate
        from .engine import DT
        if j.sol + DT >= j.deadline * self.hard or j.spent + rate * DT >= j.budget * self.hard:
            for k in range(len(j.remaining)):
                j.remaining[k] = 0.0
            self._note("backstop", f"remaining {rem:.2f} unit-sols completed to hold the cap")

    def severity(self, sev: float) -> float:
        f = min(1.0, max(0.5, 0.5 + self._slack() / 0.1))
        if f < 1:
            self._note("severity", f"incident severity {sev:.2f} -> {sev * f:.2f}")
        return sev * f

    def allow_risk(self, risk: dict) -> bool:
        if self._slack() > 0.04:
            return True
        self._note("risk", f"pending '{risk['label']}' did not fire")
        return False

    def charge(self, musd: float, what: str) -> float:
        j = self.job
        rem = sum(j.remaining)
        from .engine import DT
        steps = math.ceil(j.nominal_left(self.boost) / DT - 1e-9) if rem > 1e-9 else 0   # billed per whole step
        future = steps * DT * j.cost_rate
        room = j.budget * self.soft - j.spent - future
        amt = min(musd, max(0.0, room))
        if amt < musd - 1e-6:
            self._note("cost", f"{what}: ${musd:.2f}M charged as ${amt:.2f}M")
        return amt
