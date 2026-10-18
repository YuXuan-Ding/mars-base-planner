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


MAX_DRY_STEPS = 400_000


def estimate(plan: dict, n_units: int, n_tbms: int = 0, params: Params | None = None) -> float:
    """Sols to finish with no incidents or storms: the same schedule as the run (units, TBMs, sides of bores)."""
    d = Job(plan, n_units, params, guard=False, n_tbms=n_tbms, dry=True)
    while not d.done and d.step_i < MAX_DRY_STEPS:
        d.step()
    return d.sol


class Job:
    """Road units (U1..) grade road tiles; on routes with bores, TBMs (T1..) set up at a portal and bore
    tile by tile toward the other end, one from each end when there are two. A bore splits the road: crews
    on one side can't reach the other until it breaks through, so units start on each side (the far side
    is reached from the other base)."""

    def __init__(self, plan: dict, n_units: int, params: Params | None = None, seed: int | str = 1,
                 budget_musd: float | None = None, deadline_sols: float | None = None, guard: bool = True,
                 n_tbms: int | None = None, dry: bool = False):
        p = self.p = params or Params()
        self.plan = plan
        self.tiles = plan["tiles"]
        self.bores = [dict(b, broken=False) for b in plan.get("bores", [])]
        self.dry = dry
        self.seed = seed_int(seed)
        self.n = n_units
        self.m = (n_tbms if n_tbms is not None else 2) if self.bores else 0
        self.m = max(self.m, 1) if self.bores else 0
        self.cost_rate = n_units * p.unit_cost_musd_per_sol + self.m * p.tbm_cost_musd_per_sol
        self.road_idx = [i for i, t in enumerate(self.tiles) if t.get("kind", "road") == "road"]
        self.bore_idx = [i for i, t in enumerate(self.tiles) if t.get("kind", "road") != "road"]
        self.remaining = [t["work"] for t in self.tiles]
        self.work_total = sum(self.remaining) or 1.0
        self._par: dict = {}
        if dry:
            self.est_sols = 0.0
        else:
            self.est_sols = estimate(plan, n_units, self.m, p)
        self.est_cost = self.est_sols * self.cost_rate
        self.budget = budget_musd or round(self.est_cost * (1 + p.contingency), 2)
        self.deadline = deadline_sols or round(self.est_sols * (1 + p.contingency), 2)
        self.units = [Unit(f"U{i + 1}") for i in range(n_units)] + [Unit(f"T{i + 1}", kind="tbm") for i in range(self.m)]
        self.spares = p.spare_modules
        self.sol = 0.0
        self.step_i = 0
        self.spent = 0.0
        self.storm: dict | None = None
        self.started: set = set()
        self._fin: list = []
        self.finish_sol: float | None = None
        self.incidents: list[Incident] = []
        self.log: list[tuple[float, str]] = []
        self.frames: list[dict] = []
        self.work_rate_ema = float(n_units)       # observed road unit-sols of work per sol
        self.bore_rate_ema = float(self.m)        # observed TBM-sols of boring per sol
        self.guard = Guard(self) if guard else None
        self._place_units()
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

    @property
    def road_units(self) -> list[Unit]:
        return [u for u in self.units if u.kind == "road"]

    @property
    def tbms(self) -> list[Unit]:
        return [u for u in self.units if u.kind == "tbm"]

    # sides of the road: groups joined as bores break through

    def _find(self, g):
        if g is None:
            return None
        while self._par.get(g, g) != g:
            g = self._par[g]
        return g

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra is not None and rb is not None and ra != rb:
            self._par[ra] = rb

    def _bores_pending(self) -> bool:
        return any(not b["broken"] for b in self.bores)

    def _place_units(self):
        """Road units start on each side of the bores, in proportion to the road work there (one each first,
        biggest side first); with no bores they all start together."""
        work: dict = {}
        for i in self.road_idx:
            g = self.tiles[i].get("group")
            work[g] = work.get(g, 0.0) + self.remaining[i]
        sides = sorted(work, key=lambda g: -work[g]) if self.bores else [None]
        crews = self.road_units
        if len(sides) <= 1 or not crews:
            for u in crews:
                u.group = sides[0] if sides else None
            return
        n = len(crews)
        share = {g: 0 for g in sides}
        for g in sides[:n]:
            share[g] = 1
        left = n - sum(share.values())
        tot = sum(work[g] for g in sides) or 1.0
        while left > 0:
            g = max(sides, key=lambda g: work[g] / tot * n - share[g])
            share[g] += 1
            left -= 1
        it = iter(crews)
        for g in sides:
            for _ in range(share[g]):
                next(it).group = g

    # assignment

    def _assign(self, u: Unit):
        u.tile = self._assign_tbm(u) if u.kind == "tbm" else self._assign_road(u)
        u.rate, u.risk = 1.0, None
        if u.status in ("working", "idle"):
            u.status = "working" if u.tile is not None else "idle"

    def _assign_road(self, u: Unit):
        reach = self._find(u.group)
        open_ = [i for i in self.road_idx if self.remaining[i] > 1e-9]
        mine = [i for i in open_ if not self.bores or self._find(self.tiles[i].get("group")) == reach]
        if not mine and open_ and not self._bores_pending():
            mine = open_                      # every bore is through: the whole road is reachable
        if not mine:
            return None
        k = next((i for i in mine if i not in self.started), None)
        if k is None:
            k = max(mine, key=lambda i: self.remaining[i])
        self.started.add(k)
        u.group = self.tiles[k].get("group", u.group)
        return k

    def _face_tile(self, bi: int, side: str):
        b = self.bores[bi]
        if b["broken"]:
            return None
        portal = b["portal_a"] if side == "a" else b["portal_b"]
        if self.remaining[portal] > 1e-9:
            return portal                     # set up at the portal first
        seq = b["tiles"] if side == "a" else b["tiles"][::-1]
        return next((i for i in seq if self.remaining[i] > 1e-9), None)

    def _assign_tbm(self, u: Unit):
        if u.face is not None:
            k = self._face_tile(*u.face)
            if k is not None:
                return k
            u.face = None
        taken = {o.face for o in self.tbms if o is not u and o.face is not None}
        best = None
        for bi, b in enumerate(self.bores):
            if b["broken"]:
                continue
            faces = [f for f in ("a", "b") if (bi, f) not in taken]
            if not faces:
                continue
            left = sum(self.remaining[i] for i in b["tiles"])
            busy = 2 - len(faces)
            if best is None or (busy, -left) < best[0]:
                best = ((busy, -left), (bi, faces[0]))
        if best is None:
            return None
        u.face = best[1]
        return self._face_tile(*u.face)

    def _break_through(self, bi: int):
        b = self.bores[bi]
        b["broken"] = True
        for i in (b["portal_a"], b["portal_b"]):
            if self.remaining[i] > 0:
                self._fin.append(i)
            self.remaining[i] = 0.0           # a face never started needs no set-up
        self._union(*b["groups"])
        self._say(f"Bore {bi + 1} broke through ({b['length_m'] / 1000:.1f} km); road crews can drive through it.")
        for o in self.units:
            if o.kind == "tbm" and o.face and o.face[0] == bi:
                o.face = None
            if o.status in ("working", "idle") and (o.tile is None or self.remaining[o.tile] <= 1e-9):
                self._assign(o)

    def _charge(self, musd: float, what: str) -> float:
        amt = self.guard.charge(musd, what) if self.guard else musd
        self.spent += amt
        return amt

    # projections, as the planner would compute them from what it can see

    def road_left(self) -> float:
        return sum(self.remaining[i] for i in self.road_idx)

    def bore_left(self) -> float:
        return sum(self.remaining[i] for i in self.bore_idx)

    def nominal_left(self, boost: float = 1.0) -> float:
        """Sols of work left at the rates of the machines working now (the road and the bores in parallel)."""
        storm = 1.0
        if self.storm:
            storm = 0.5 if self.storm["mode"] == "work_through" else 0.0 if self.storm["mode"] else 0.5
        road_a = max(sum(u.rate for u in self.road_units if u.status == "working") * storm, 0.5) * boost
        bore_a = max(sum(u.rate for u in self.tbms if u.status == "working"), 0.5) * boost
        bl = self.bore_left()
        return max(self.road_left() / road_a, bl / bore_a if bl > 1e-9 else 0.0)

    def projection(self) -> dict:
        finish = self.sol + max(self.road_left() / max(self.work_rate_ema, 0.05),
                                self.bore_left() / max(self.bore_rate_ema, 0.05) if self.bore_left() > 1e-9 else 0.0)
        cost = self.spent + (finish - self.sol) * self.cost_rate
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
        elif not self.storm and not self.dry and u01(s, k, CH_STORM) < p.storm_per_sol * DT * (g.incident_mult if g else 1):
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
        done_now = bored_now = 0.0
        self._fin = []
        for ui, u in enumerate(self.units):
            if u.status != "working" or u.tile is None:
                continue
            ti = u.tile
            under = u.kind == "tbm"            # TBMs work underground: storms don't slow them
            sr, rk = (1.0, 1.0) if under else (storm_rate, storm_risk)
            w = min(self.remaining[ti], u.rate * sr * boost * DT)
            self.remaining[ti] -= w
            u.worked += w
            if under:
                bored_now += w
            else:
                done_now += w
            exposure = u.rate * sr * DT
            lam = self.tiles[ti]["incident_rate"] * rk * (g.incident_mult if g else 1.0)
            if not self.dry and exposure > 0 and u01(s, k, CH_INCIDENT, ui) < 1 - math.exp(-lam * exposure):
                self._open_tile_incident(u, ui)
            if u.risk and u.status == "working" and exposure > 0:
                if u01(s, k, CH_RISK, ui) < 1 - math.exp(-u.risk["p_per_sol"] * DT):
                    self._fire_risk(u)
            if self.remaining[ti] <= 1e-9:
                self.remaining[ti] = 0.0
                self._fin.append(ti)
                tk = self.tiles[ti].get("kind", "road")
                if not self.dry:
                    what = {"portal": "Portal set up", "bore": "Bored through tile"}.get(tk, "Tile")
                    self._say(f"{what} {ti} finished ({self.tiles[ti]['lon']:.3f}E {self.tiles[ti]['lat']:.3f}N).")
                bi = self.tiles[ti].get("bore")
                if tk == "bore" and not self.bores[bi]["broken"] and all(
                        self.remaining[i] <= 1e-9 for i in self.bores[bi]["tiles"]):
                    self._break_through(bi)
                if u.status == "working":
                    self._assign(u)
                for o in self.units:
                    if o is not u and o.tile == ti and o.status == "working":
                        self._assign(o)
        for u in self.units:                   # idle crews pick up work that became reachable
            if u.status == "idle":
                self._assign(u)

        self.work_rate_ema = 0.85 * self.work_rate_ema + 0.15 * (done_now / DT)
        self.bore_rate_ema = 0.85 * self.bore_rate_ema + 0.15 * (bored_now / DT)
        self.spent += self.cost_rate * DT
        self.sol = round(self.sol + DT, 4)
        self.step_i += 1
        if g:
            g.after_step()
        if all(r <= 1e-9 for r in self.remaining) and self.finish_sol is None:
            self.finish_sol = self.sol
            for u in self.units:
                u.status, u.tile = "done", None
            self._say("Job complete.")
        if not self.dry:
            self._frame()

    def _frame(self):
        self.frames.append({
            "sol": self.sol, "progress_pct": round(100 * (1 - sum(self.remaining) / self.work_total), 2),
            "spent_musd": round(self.spent, 3), "storm": bool(self.storm),
            "units": [{"id": u.id, "kind": u.kind, "status": u.status, "tile": u.tile} for u in self.units],
            "tiles_left": sum(r > 1e-9 for r in self.remaining),
            "finished": list(getattr(self, "_fin", [])),     # tiles completed in this step
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
        if u.kind == "tbm":
            return self._open_tbm_incident(u, sev)
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

    def _open_tbm_incident(self, u: Unit, sev: float):
        tile = self.tiles[u.tile]
        bi = tile.get("bore", 0)
        title = f"{u.id} cutterhead fault"
        report = (f"{u.id}'s cutterhead torque spiked boring bore {bi + 1} under {tile['lon']:.3f}E {tile['lat']:.3f}N "
                  f"(rock cover unknown; surface hazard {tile['hazard']:.2f}). Severity {sev:.1f}.")
        opts = [Option("change_cutters", "Stop and change the worn cutters", {"down": 1.0 * sev, "cost": 0.4 * sev}),
                Option("slow_bore", "Bore on at 60% thrust until this stretch is done",
                       {"rate": 0.6, "risk": {"p_per_sol": 0.2 * sev, "down": 5.0, "cost": 1.2, "label": "main bearing damage"}},
                       risk=f"{0.2 * sev:.0%}/sol chance of main bearing damage (5 sols down, $1.2M)")]
        u.status, u.down_reason, u.down_until = "down", "incident", float("inf")
        inc = Incident(f"I{len(self.incidents) + 1}", "tbm_fault", self.sol, u.id, u.tile, sev, title, report)
        inc.options = [self._estimate_option(o, u) for o in opts]
        self.incidents.append(inc)
        self._say(f"INCIDENT {inc.id}: {title}")

    def _open_storm(self):
        st = self.storm
        working = sum(u.status == "working" for u in self.road_units)
        inc = Incident(f"I{len(self.incidents) + 1}", "dust_storm", self.sol, None, None, 1.0, "Regional dust storm",
                       f"Visibility dropping fast; {working} units in the field. Storms here last "
                       f"{self.p.storm_len_sols[0]}-{self.p.storm_len_sols[1]} sols. Duration unknown.")
        mid = sum(self.p.storm_len_sols) / 2
        inc.options = [
            Option("pause", "Pause all surface units until it clears", {"storm": "pause"}, est_delay_sols=mid,
                   est_cost_musd=mid * self.cost_rate),
            Option("work_through", "Work through it at half speed", {"storm": "work_through"}, est_delay_sols=mid / 2,
                   est_cost_musd=mid / 2 * self.cost_rate,
                   risk="incident odds x2.5 while the storm lasts"),
            Option("service", "Pause and service every surface unit (clears degraded running)", {"storm": "service", "cost": 0.2},
                   est_delay_sols=mid, est_cost_musd=mid * self.cost_rate + 0.2),
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
        delay = lost / (max(self.m, 1) if u.kind == "tbm" else self.n)   # a TBM fault holds up its bore directly
        cost = e.get("cost", 0) + (pf * r["cost"] if r else 0) + delay * self.cost_rate
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
                for u in self.road_units:
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
            "machines": {"units": self.n, "tbms": self.m},
            "bores": [{"index": i + 1, "length_m": b["length_m"], "broken_through": b["broken"],
                       "bored_pct": round(100 * (1 - sum(self.remaining[k] for k in b["tiles"])
                                                 / (sum(self.tiles[k]["work"] for k in b["tiles"]) or 1)), 1)}
                      for i, b in enumerate(self.bores)],
            "units": [{"id": u.id, "kind": u.kind, "status": u.status, "reason": u.down_reason or None, "tile": u.tile,
                       "lonlat": [self.tiles[u.tile]["lon"], self.tiles[u.tile]["lat"]] if u.tile is not None else None,
                       "rate": u.rate, "back_sol": None if u.status != "down" or math.isinf(u.down_until) else round(u.down_until, 2),
                       "running_degraded": u.risk is not None} for u in self.units],
            "tiles_left": sum(r > 1e-9 for r in self.remaining),
            "open_incident": self.incident_view(self.open_incident) if self.open_incident else None,
        }

    def report(self) -> dict:
        fin = self.finish_sol
        return {
            "kind": self.plan["kind"], "units": self.n, "tbms": self.m, "done": self.done,
            "planned": {"sols": round(self.est_sols, 2), "cost_musd": round(self.est_cost, 2)},
            "target": {"deadline_sols": self.deadline, "budget_musd": self.budget},
            "actual": {"sols": fin, "cost_musd": round(self.spent, 2)},
            "overrun_pct": {"time": round(100 * (fin / self.deadline - 1), 1) if fin else None,
                            "cost": round(100 * (self.spent / self.budget - 1), 1)},
            "incidents": [self.incident_view(i) for i in self.incidents],
        }
