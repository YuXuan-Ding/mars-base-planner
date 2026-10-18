"""Construction simulator: units excavate tiles, tiles raise incidents, a decider picks a response.

Time moves in steps of `DT` sols. Each working unit chips away at the work on its tile.
Every unit-sol of work on a tile carries that tile's incident odds (hazard and slope), so
rocky, steep tiles cause more trouble. When an incident opens the job pauses until a
decision (the AI's, a fallback rule's, or a person's) is applied.

A guard (guard.py) runs underneath and keeps the job within its overrun cap. It is not
part of anything the decider sees.
"""
from __future__ import annotations

import math
import zlib

from .guard import Guard
from .model import CH_INCIDENT, CH_KIND, CH_RISK, CH_SEV, CH_STORM, CH_STORM_LEN, Incident, Option, Params, Unit, u01

DT = 0.25   # sols per step (one 6-hour shift)
KIND_ORDER = ("boulder", "slope_failure", "breakdown", "stuck")


def seed_int(seed: int | str) -> int:
    return seed if isinstance(seed, int) else zlib.crc32(str(seed).encode())


def estimate(tiles: list, n_units: int) -> float:
    """Sols to finish with no incidents, using the same tile-by-tile assignment as the run."""
    rem = [t["work"] for t in tiles]
    unit_tile = [None] * n_units
    nxt, sol = 0, 0.0
    while any(r > 1e-9 for r in rem):
        for i in range(n_units):
            if unit_tile[i] is None or rem[unit_tile[i]] <= 1e-9:
                if nxt < len(rem):
                    unit_tile[i], nxt = nxt, nxt + 1
                else:
                    open_ = [k for k, r in enumerate(rem) if r > 1e-9]
                    unit_tile[i] = max(open_, key=lambda k: rem[k]) if open_ else None
            k = unit_tile[i]
            if k is not None:
                rem[k] = max(0.0, rem[k] - DT)
        sol += DT
    return sol


class Job:
    def __init__(self, plan: dict, n_units: int, params: Params | None = None, seed: int | str = 1,
                 budget_musd: float | None = None, deadline_sols: float | None = None, guard: bool = True):
        p = self.p = params or Params()
        self.plan = plan
        self.tiles = plan["tiles"]
        self.seed = seed_int(seed)
        self.n = n_units
        self.remaining = [t["work"] for t in self.tiles]
        self.work_total = sum(self.remaining)
        self.est_sols = estimate(self.tiles, n_units)
        self.est_cost = self.est_sols * n_units * p.unit_cost_musd_per_sol
        self.budget = budget_musd or round(self.est_cost * (1 + p.contingency), 2)
        self.deadline = deadline_sols or round(self.est_sols * (1 + p.contingency), 2)
        self.units = [Unit(f"U{i + 1}") for i in range(n_units)]
        self.spares = p.spare_modules
        self.sol = 0.0
        self.step_i = 0
        self.spent = 0.0
        self.storm: dict | None = None
        self.next_tile = 0
        self.finish_sol: float | None = None
        self.incidents: list[Incident] = []
        self.log: list[tuple[float, str]] = []
        self.frames: list[dict] = []
        self.work_rate_ema = float(n_units)       # observed unit-sols of work per sol
        self.guard = Guard(self) if guard else None
        for u in self.units:
            self._assign(u)

    # helpers

    @property
    def done(self) -> bool:
        return self.finish_sol is not None

    @property
    def open_incident(self) -> Incident | None:
        return next((i for i in self.incidents if i.status == "open"), None)

    def unit(self, uid: str) -> Unit:
        return next(u for u in self.units if u.id == uid)

    def _say(self, text: str):
        self.log.append((round(self.sol, 2), text))

    def _assign(self, u: Unit):
        if self.next_tile < len(self.tiles):
            u.tile, self.next_tile = self.next_tile, self.next_tile + 1
        else:
            open_ = [k for k, r in enumerate(self.remaining) if r > 1e-9]
            u.tile = max(open_, key=lambda k: self.remaining[k]) if open_ else None
        u.rate, u.risk = 1.0, None
        if u.status in ("working", "idle"):
            u.status = "working" if u.tile is not None else "idle"

    def _charge(self, musd: float, what: str) -> float:
        amt = self.guard.charge(musd, what) if self.guard else musd
        self.spent += amt
        return amt

    # projections, as the planner would compute them from what it can see

    def projection(self) -> dict:
        rem = sum(self.remaining)
        rate = max(self.work_rate_ema, 0.05)
        finish = self.sol + rem / rate
        cost = self.spent + (finish - self.sol) * self.n * self.p.unit_cost_musd_per_sol
        return {"finish_sol": round(finish, 1), "cost_musd": round(cost, 2),
                "time_overrun_pct": round(100 * (finish / self.deadline - 1), 1),
                "cost_overrun_pct": round(100 * (cost / self.budget - 1), 1)}

    # one step

    def step(self):
        if self.done or self.open_incident:
            return
        p, s, k = self.p, self.seed, self.step_i
        g = self.guard
        if g:
            g.before_step()

        # weather
        if self.storm and self.sol >= self.storm["end"]:
            self._say("Dust storm cleared.")
            self.storm = None
            for u in self.units:
                if u.status == "down" and u.down_reason == "sheltering":
                    u.status, u.down_reason = "working", ""
                    if u.tile is None or self.remaining[u.tile] <= 1e-9:
                        self._assign(u)
        elif not self.storm and u01(s, k, CH_STORM) < p.storm_per_sol * DT * (g.incident_mult if g else 1):
            lo, hi = p.storm_len_sols
            ln = lo + u01(s, k, CH_STORM_LEN) * (hi - lo)
            self.storm = {"start": self.sol, "end": self.sol + ln, "mode": None, "factor": 0.5}
            self._open_storm()
            return

        for u in self.units:
            if u.status == "down" and u.down_reason != "sheltering" and self.sol >= u.down_until:
                u.status, u.down_reason = "working", ""
                self._say(f"{u.id} back at work.")
                if u.tile is None or self.remaining[u.tile] <= 1e-9:
                    self._assign(u)

        storm_rate, storm_risk = 1.0, 1.0
        if self.storm:
            storm_rate = {"pause": 0.0, "work_through": 0.5, "service": 0.0}.get(self.storm["mode"], 0.5)
            storm_risk = 2.5 if self.storm["mode"] == "work_through" else 1.0
        boost = g.boost if g else 1.0
        done_now = 0.0
        for ui, u in enumerate(self.units):
            if u.status != "working" or u.tile is None:
                continue
            ti = u.tile
            w = min(self.remaining[ti], u.rate * storm_rate * boost * DT)
            self.remaining[ti] -= w
            u.worked += w
            done_now += w
            exposure = u.rate * storm_rate * DT
            lam = self.tiles[ti]["incident_rate"] * storm_risk * (g.incident_mult if g else 1.0)
            if exposure > 0 and u01(s, k, CH_INCIDENT, ui) < 1 - math.exp(-lam * exposure):
                self._open_tile_incident(u, ui)
            if u.risk and u.status == "working" and exposure > 0:
                if u01(s, k, CH_RISK, ui) < 1 - math.exp(-u.risk["p_per_sol"] * DT):
                    self._fire_risk(u)
            if self.remaining[ti] <= 1e-9:
                self.remaining[ti] = 0.0
                self._say(f"Tile {ti} finished ({self.tiles[ti]['lon']:.3f}E {self.tiles[ti]['lat']:.3f}N).")
                if u.status == "working":
                    self._assign(u)
                for o in self.units:
                    if o is not u and o.tile == ti and o.status == "working":
                        self._assign(o)

        self.work_rate_ema = 0.85 * self.work_rate_ema + 0.15 * (done_now / DT)
        self.spent += self.n * p.unit_cost_musd_per_sol * DT
        self.sol = round(self.sol + DT, 4)
        self.step_i += 1
        if g:
            g.after_step()
        if all(r <= 1e-9 for r in self.remaining) and self.finish_sol is None:
            self.finish_sol = self.sol
            for u in self.units:
                u.status, u.tile = "done", None
            self._say("Job complete.")
        self._frame()

    def _frame(self):
        self.frames.append({
            "sol": self.sol, "progress_pct": round(100 * (1 - sum(self.remaining) / self.work_total), 2),
            "spent_musd": round(self.spent, 3), "storm": bool(self.storm),
            "units": [{"id": u.id, "status": u.status, "tile": u.tile} for u in self.units],
            "tiles_left": sum(r > 1e-9 for r in self.remaining),
        })

    # incidents

    def _open_tile_incident(self, u: Unit, ui: int):
        p, s, k = self.p, self.seed, self.step_i
        tile = self.tiles[u.tile]
        x, acc, kind = u01(s, k, CH_KIND, ui), 0.0, "breakdown"
        for name in KIND_ORDER:
            acc += tile["kinds"][name]
            if x < acc:
                kind = name
                break
        sev = 0.5 + u01(s, k, CH_SEV, ui)
        if self.guard:
            sev = self.guard.severity(sev)
        sev = round(sev, 2)
        where = f"tile {u.tile} ({tile['lon']:.3f}E {tile['lat']:.3f}N; slope {tile['slope_deg']:.1f} deg, " \
                f"rock cover {tile['rock_cover_pct']:.1f}%, hazard {tile['hazard']:.2f})"
        opts: list[Option] = []
        if kind == "breakdown":
            title, report = f"{u.id} hydraulic failure", f"{u.id} lost hydraulic pressure on {where}. Severity {sev:.1f}."
            opts.append(Option("repair", "Repair on site", {"down": 1.5 * sev, "cost": 0.5 * sev}))
            if self.spares > 0:
                opts.append(Option("swap", f"Swap in a spare drive module ({self.spares} left)",
                                   {"down": 0.25, "cost": 0.25, "spare": 1}))
            opts.append(Option("limp", "Keep working at 60% until this tile is done",
                               {"rate": 0.6, "risk": {"p_per_sol": 0.25 * sev, "down": 4.0, "cost": 0.8, "label": "drive failure"}},
                               risk=f"{0.25 * sev:.0%}/sol chance the drive fails outright (4 sols down, $0.8M)"))
        elif kind == "boulder":
            title, report = f"{u.id} hit a boulder field", f"{u.id} hit buried boulders on {where}. Severity {sev:.1f}."
            opts.append(Option("blast", "Blast and clear", {"down": 0.5, "cost": 0.3 * sev}))
            opts.append(Option("work_around", "Dig around the boulders", {"work_mult": 1 + 0.35 * sev}))
            if self.plan["kind"] == "route":
                opts.append(Option("realign", "Shift the alignment off the boulders (survey + regrade)",
                                   {"work_mult": 1 + 0.15 * sev, "cost": 0.15}))
        elif kind == "slope_failure":
            title, report = f"Slope failure on {u.id}'s cut", f"The cut slumped under {u.id} on {where}. Severity {sev:.1f}."
            opts.append(Option("shore", "Shore the cut and regrade", {"down": 0.5, "cost": 0.6 * sev, "work_mult": 1 + 0.2 * sev}))
            opts.append(Option("lay_back", "Lay the slope back (cut wider, slower)", {"work_mult": 1 + 0.7 * sev, "cost": 0.05}))
            opts.append(Option("push_on", "Clear the slump and push on",
                               {"work_mult": 1.05, "risk": {"p_per_sol": 0.35, "down": 3 * sev, "cost": 0.8 * sev,
                                                            "work_mult": 1.3, "label": "second collapse"}},
                               risk=f"35%/sol chance of a second collapse ({3 * sev:.1f} sols down, ${0.8 * sev:.1f}M)"))
        else:
            title, report = f"{u.id} stuck in loose regolith", f"{u.id} bogged down in soft ground on {where}. Severity {sev:.1f}."
            helpers = [o for o in self.units if o is not u and o.status == "working"]
            if helpers:
                opts.append(Option("tow", f"Tow out with {helpers[0].id}", {"down": 0.25 * sev, "helper": helpers[0].id,
                                                                           "helper_down": 0.5 * sev}))
            opts.append(Option("recovery_kit", "Wait for the recovery kit from base", {"down": 1.5 * sev, "cost": 0.1}))
        u.status, u.down_reason, u.down_until = "down", "incident", float("inf")
        inc = Incident(f"I{len(self.incidents) + 1}", kind, self.sol, u.id, u.tile, sev, title, report)
        inc.options = [self._estimate_option(o, u) for o in opts]
        self.incidents.append(inc)
        self._say(f"INCIDENT {inc.id}: {title}")

    def _open_storm(self):
        st = self.storm
        working = sum(u.status == "working" for u in self.units)
        inc = Incident(f"I{len(self.incidents) + 1}", "dust_storm", self.sol, None, None, 1.0, "Regional dust storm",
                       f"Visibility dropping fast; {working} units in the field. Storms here last "
                       f"{self.p.storm_len_sols[0]}-{self.p.storm_len_sols[1]} sols. Duration unknown.")
        mid = sum(self.p.storm_len_sols) / 2
        inc.options = [
            Option("pause", "Pause all units until it clears", {"storm": "pause"}, est_delay_sols=mid,
                   est_cost_musd=mid * self.n * self.p.unit_cost_musd_per_sol),
            Option("work_through", "Work through it at half speed", {"storm": "work_through"}, est_delay_sols=mid / 2,
                   est_cost_musd=mid / 2 * self.n * self.p.unit_cost_musd_per_sol,
                   risk="incident odds x2.5 while the storm lasts"),
            Option("service", "Pause and service every unit (clears degraded running)", {"storm": "service", "cost": 0.2},
                   est_delay_sols=mid, est_cost_musd=mid * self.n * self.p.unit_cost_musd_per_sol + 0.2),
        ]
        self.incidents.append(inc)
        self._say(f"INCIDENT {inc.id}: dust storm")

    def _estimate_option(self, o: Option, u: Unit) -> Option:
        """Nominal effect on finish and cost, as a planner would reckon it."""
        p, e = self.p, o.effects
        rem_tile = self.remaining[u.tile] if u.tile is not None else 0.0
        lost = e.get("down", 0) + e.get("helper_down", 0) + (e.get("work_mult", 1) - 1) * rem_tile
        if "rate" in e:
            lost += rem_tile * (1 / e["rate"] - 1)
        r = e.get("risk")
        if r:
            exposure = rem_tile / e.get("rate", 1.0)
            pf = 1 - math.exp(-r["p_per_sol"] * exposure)
            lost += pf * (r["down"] + (r.get("work_mult", 1) - 1) * rem_tile)
        delay = lost / self.n
        cost = e.get("cost", 0) + (pf * r["cost"] if r else 0) + delay * self.n * p.unit_cost_musd_per_sol
        o.est_delay_sols, o.est_cost_musd = round(delay, 2), round(cost, 2)
        return o

    def decide(self, incident_id: str, option_id: str, by: str, rationale: str = "") -> dict:
        inc = next((i for i in self.incidents if i.id == incident_id), None)
        if inc is None:
            raise KeyError(f"no incident {incident_id}")
        if inc.status != "open":
            raise ValueError(f"incident {incident_id} is already resolved")
        opt = next((o for o in inc.options if o.id == option_id), None)
        if opt is None:
            raise ValueError(f"option must be one of {[o.id for o in inc.options]}")
        inc.status, inc.choice, inc.decided_by, inc.rationale = "resolved", option_id, by, rationale
        e = opt.effects
        if "storm" in e:
            self.storm["mode"] = e["storm"]
            if e["storm"] in ("pause", "service"):
                for u in self.units:
                    if u.status == "working":
                        u.status, u.down_reason = "down", "sheltering"
                    if e["storm"] == "service":
                        u.rate, u.risk = 1.0, None
            if e.get("cost"):
                self._charge(e["cost"], "unit servicing")
            inc.outcome = f"Storm response: {opt.label.lower()}."
        else:
            u = self.unit(inc.unit)
            u.status, u.down_reason = "working", ""
            if e.get("down"):
                u.status, u.down_reason, u.down_until = "down", inc.kind, self.sol + e["down"]
            if e.get("cost"):
                self._charge(e["cost"], f"{inc.kind} response")
            if e.get("spare"):
                self.spares -= e["spare"]
            if e.get("work_mult") and u.tile is not None:
                self.remaining[u.tile] *= e["work_mult"]
            if e.get("rate"):
                u.rate = e["rate"]
            if e.get("risk"):
                u.risk = dict(e["risk"])
            if e.get("helper"):
                h = self.unit(e["helper"])
                if h.status == "working":
                    h.status, h.down_reason, h.down_until = "down", "towing", self.sol + e["helper_down"]
            inc.outcome = f"{opt.label}."
        self._say(f"{inc.id}: {opt.label} (decided by {by}).")
        return {"incident": inc.id, "option": option_id}

    def _fire_risk(self, u: Unit):
        r = u.risk
        if self.guard and not self.guard.allow_risk(r):
            u.risk = None
            return
        u.risk = None
        u.status, u.down_reason, u.down_until = "down", r["label"], self.sol + r["down"]
        self._charge(r["cost"], r["label"])
        if r.get("work_mult") and u.tile is not None:
            self.remaining[u.tile] *= r["work_mult"]
        self._say(f"{u.id}: {r['label']} ({r['down']:.1f} sols down).")
        for inc in reversed(self.incidents):
            if inc.unit == u.id:
                inc.outcome += f" Then: {r['label']} on sol {self.sol:.1f}."
                break

    # running

    def run(self, max_sols: float | None = None, decider=None) -> list[Incident]:
        """Fast-forward. Stops at the first open incident unless `decider(job, incident)`
        is given, in which case it decides each one and keeps going."""
        stop = self.sol + max_sols if max_sols else math.inf
        handled = []
        while not self.done and self.sol < stop:
            inc = self.open_incident
            if inc:
                if decider is None:
                    break
                choice, by, why = decider(self, inc)
                self.decide(inc.id, choice, by, why)
                handled.append(inc)
                continue
            self.step()
        return handled

    # views (no guard internals)

    def option_view(self, o: Option) -> dict:
        return {"id": o.id, "label": o.label, "est_delay_sols": o.est_delay_sols, "est_cost_musd": o.est_cost_musd,
                "risk": o.risk or None}

    def incident_view(self, inc: Incident) -> dict:
        d = {"id": inc.id, "kind": inc.kind, "sol": inc.sol, "unit": inc.unit, "title": inc.title,
             "report": inc.report, "severity": inc.severity, "status": inc.status,
             "options": [self.option_view(o) for o in inc.options]}
        if inc.tile is not None:
            t = self.tiles[inc.tile]
            d["tile"] = {"index": inc.tile, "lon": t["lon"], "lat": t["lat"], "slope_deg": t["slope_deg"],
                         "rock_cover_pct": t["rock_cover_pct"], "hazard": t["hazard"],
                         "work_left_unit_sols": round(self.remaining[inc.tile], 2)}
        if inc.status == "resolved":
            d.update(choice=inc.choice, decided_by=inc.decided_by, rationale=inc.rationale, outcome=inc.outcome)
        return d

    def state(self) -> dict:
        return {
            "sol": self.sol, "done": self.done, "finish_sol": self.finish_sol,
            "progress_pct": round(100 * (1 - sum(self.remaining) / self.work_total), 2),
            "spent_musd": round(self.spent, 3),
            "target": {"budget_musd": self.budget, "deadline_sols": self.deadline,
                       "estimate_sols": round(self.est_sols, 2), "estimate_cost_musd": round(self.est_cost, 2)},
            "projection": self.projection(),
            "storm": {"since_sol": self.storm["start"], "response": self.storm["mode"]} if self.storm else None,
            "spares": self.spares,
            "units": [{"id": u.id, "status": u.status, "reason": u.down_reason or None, "tile": u.tile,
                       "lonlat": [self.tiles[u.tile]["lon"], self.tiles[u.tile]["lat"]] if u.tile is not None else None,
                       "rate": u.rate, "back_sol": None if u.status != "down" or math.isinf(u.down_until) else round(u.down_until, 2),
                       "running_degraded": u.risk is not None} for u in self.units],
            "tiles_left": sum(r > 1e-9 for r in self.remaining),
            "open_incident": self.incident_view(self.open_incident) if self.open_incident else None,
        }

    def report(self) -> dict:
        fin = self.finish_sol
        return {
            "kind": self.plan["kind"], "units": self.n, "done": self.done,
            "planned": {"sols": round(self.est_sols, 2), "cost_musd": round(self.est_cost, 2)},
            "target": {"deadline_sols": self.deadline, "budget_musd": self.budget},
            "actual": {"sols": fin, "cost_musd": round(self.spent, 2)},
            "overrun_pct": {"time": round(100 * (fin / self.deadline - 1), 1) if fin else None,
                            "cost": round(100 * (self.spent / self.budget - 1), 1)},
            "incidents": [self.incident_view(i) for i in self.incidents],
        }
