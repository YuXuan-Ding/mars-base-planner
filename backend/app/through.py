"""Bore through: one road along the straight A–B line, bored level wherever the ground rises above a chosen height.

The route itself never changes: it is the straight line from A to B, on the grid cells it crosses. The only
decision is the **bore height**. Every stretch where the ground is above it is a level TBM bore at that height
(portals where the ground crosses it); everything else is a built road on the line. A lower height means less
climbing and descending on the road and a longer bore; a height at or above the highest ground means no bore.

    score = road + each bore + resurface_penalty_sols for each bore after the first
    road  = the shortest route's move costs along the line (length, hazards, climb, grading), in metres of driving
    bore  = its boring sols (advance + set-up) as metres of driving + its length
            + tunnel_vertical_m × the height its portal ramps make up

The height is picked by the rule here (`Through.best`: the lowest score over a scan of heights) or by Claude
(`plan_with_claude`), who reads the elevation along the line and tries heights before submitting one.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.sparse.csgraph import dijkstra

from . import tunnels
from .routing import BORE_SETUP_SOLS, HORIZON_SOLS, _build_road, _drive_sols_per_m, _moves, _road_graph

MIN_BORE_M = 300          # shorter humps are driven over: not worth a TBM launch
SCAN_LEVELS = 80          # heights the rule tries between the higher end and the highest ground
LINE_ROWS = 200           # rows in the line profile returned for the chart


class Through:
    def __init__(self, ctx, a: int, b: int, req, tbm):
        self.ctx, self.a, self.b, self.req, self.raw_tbm = ctx, a, b, req, tbm
        t, g = ctx.t, ctx.g
        tg = math.tan(math.radians(ctx.fleet.limits.max_grade_deg))
        self.tbm = tbm.model_copy(update={"max_grade_pct": min(tbm.max_grade_pct, 100 * tg)})
        self.G = _road_graph(ctx, req)
        self.drive_m = _drive_sols_per_m(ctx, req)
        self.rate = tunnels.bore_rate(tbm) * min(req.tbm.count, 2)
        # the line, on the grid cells it crosses (each step to one of the 8 neighbours)
        C = g.shape[1]
        (ra, ca), (rb, cb) = divmod(a, C), divmod(b, C)
        k = max(abs(rb - ra), abs(cb - ca), 1)
        rr = np.rint(np.linspace(ra, rb, k + 1)).astype(int)
        cc = np.rint(np.linspace(ca, cb, k + 1)).astype(int)
        nodes = rr * C + cc
        self.nodes = nodes[np.concatenate([[True], np.diff(nodes) != 0])]
        m = _moves(ctx, self.nodes)
        self.cum = np.concatenate([[0.0], np.cumsum(m["L"])])
        self.H = g.H.ravel()[self.nodes].astype(float)
        self.slope = g.slope.ravel()[self.nodes].astype(float)
        u, v = self.nodes[:-1], self.nodes[1:]
        mc = np.asarray(self.G[u, v]).ravel() + np.asarray(self.G[v, u]).ravel()   # moves are stored one way
        self.move_cost = np.where(mc > 0, mc, np.inf)                                # missing: a cliff
        R0, C0 = ctx.rc(np.array([a, b]))
        A, B = t.lonlat(R0[0], C0[0]), t.lonlat(R0[1], C0[1])
        self.A, self.B = (float(A[0]), float(A[1])), (float(B[0]), float(B[1]))
        self.lo = float(max(self.H[0], self.H[-1]))     # a bore can't go below either end
        self.hi = float(self.H.max())                   # at or above this: no bore
        self.shortest_m = float(dijkstra(self.G, directed=False, indices=a)[b])

    def km(self, i: int) -> float:
        return round(float(self.cum[i]) / 1000, 2)

    # ---------- a height ----------
    def _runs(self, h: float) -> list[tuple[int, int]]:
        """Bores at height h: (portal before, portal after) around each run of ground above h, MIN_BORE_M or more."""
        above = self.H > h
        out, i, n = [], 0, len(self.H)
        while i < n:
            if not above[i]:
                i += 1
                continue
            j = i
            while j < n and above[j]:
                j += 1
            p, q = max(i - 1, 0), min(j, n - 1)
            if self.cum[q] - self.cum[p] >= MIN_BORE_M:
                out.append((p, q))
            i = j
        return out

    def _bore_cost(self, p: int, q: int, h: float):
        L = float(self.cum[q] - self.cum[p])
        vert = abs(self.H[p] - h) + abs(self.H[q] - h)       # portal ramps from the ground to the level bore
        sols = L / self.rate + BORE_SETUP_SOLS
        return L, vert, sols, sols / self.drive_m + L + self.req.tunnel_vertical_m * vert

    def score(self, h: float | None) -> dict:
        """Fast score for a height (no road design): feasibility, score_sols, bored km."""
        req = self.req
        if h is not None and h < self.lo - 0.5:
            return {"ok": False, "reason": f"{h:.0f} m is below the ground at {'A' if self.H[0] > self.H[-1] else 'B'} "
                                           f"({self.lo:.0f} m): the lowest bore height is {self.lo:.0f} m"}
        runs = [] if h is None or h >= self.hi else self._runs(h)
        road = np.ones(len(self.move_cost), bool)
        for p, q in runs:
            road[p:q] = False
            if self.cum[q] - self.cum[p] > req.max_tunnel_km * 1000:
                return {"ok": False, "reason": f"the bore from km {self.km(p)} to km {self.km(q)} is longer than the "
                                               f"{req.max_tunnel_km:g} km tunnel limit"}
        rc = self.move_cost[road]
        if not np.isfinite(rc).all():
            k = int(np.flatnonzero(road)[np.argmax(~np.isfinite(rc))])
            top = max(self.H[k], self.H[k + 1])
            return {"ok": False, "reason": f"the road crosses a cliff at km {self.km(k)} (top at {top:.0f} m); "
                                           f"bore below {top:.0f} m to tunnel past it"}
        cost_m = float(rc.sum()) + sum(self._bore_cost(p, q, h)[3] for p, q in runs)
        return {"ok": True, "runs": runs, "tunnel_m": sum(float(self.cum[q] - self.cum[p]) for p, q in runs),
                "score_sols": cost_m * self.drive_m + req.resurface_penalty_sols * max(0, len(runs) - 1)}

    def evaluate(self, h: float | None) -> dict:
        """The whole plan for bore height h (None: no bore): roads, bores, totals and score."""
        ctx, req = self.ctx, self.req
        if h is not None and h >= self.hi:
            h = None
        sc = self.score(h)
        if not sc["ok"]:
            return {"status": "none", "reason": sc["reason"], "height_m": h}
        runs, bores = sc["runs"], []
        for p, q in runs:
            L, vert, sols, _ = self._bore_cost(p, q, h)
            s = self.cum[p:q + 1] - self.cum[p]
            surf = self.H[p:q + 1]
            z = np.full_like(surf, h)
            r_, c_ = ctx.rc(self.nodes[[p, q]])
            lon, lat = ctx.t.lonlat(r_, c_)
            prof = {"s": s, "surf": surf, "z": z}
            idx = np.unique(np.linspace(0, len(s) - 1, min(len(s), 60)).round().astype(int))
            bores.append({
                "from_km": self.km(p), "to_km": self.km(q), "height_m": round(h), "length_km": round(L / 1000, 2),
                "vertical_m": round(vert), "bore_sols": round(sols, 1), "setup_sols": BORE_SETUP_SOLS,
                "spoil_m3": round(tunnels.face_area(self.tbm) * L * self.tbm.swell),
                "cover": {k: v for k, v in tunnels._check(s, surf, z, self.tbm).items()
                          if k in ("min_cover_m", "max_cover_m", "shallow_m", "portal_cut_m")},
                "geometry": {"type": "LineString", "coordinates": [[round(float(lon[0]), 5), round(float(lat[0]), 5)],
                                                                   [round(float(lon[1]), 5), round(float(lat[1]), 5)]]},
                "profile": [[round(float(s[i]) / 1000, 3), round(float(surf[i]), 1), round(float(h), 1)] for i in idx],
                "feasible": True})
        stops = [0] + [k for pq in runs for k in pq] + [len(self.nodes) - 1]
        legs = [_build_road(ctx, self.nodes[x:y + 1], req, self.raw_tbm)
                for x, y in zip(stops[::2], stops[1::2]) if y > x]
        tun_m = sc["tunnel_m"]
        road_km = sum(l["length_km"] for l in legs)
        length_km = road_km + tun_m / 1000
        grading = sum(l["grading_sols"] for l in legs)
        clearing = sum(l["clearing_sols"] for l in legs)
        bore_sols = sum(b["bore_sols"] for b in bores) + sum(l.get("bore_sols", 0) for l in legs)
        no_bore = self.shortest_m * self.drive_m
        return {
            "status": "ok", "height_m": None if h is None else round(h),
            "bores": bores, "legs": [_leg_view(l) for l in legs],
            "length_km": round(length_km, 2), "road_km": round(road_km, 2), "tunnel_km": round(tun_m / 1000, 2),
            "travel_h": round(length_km / ctx.fleet.slowest_kmh, 1),
            "climb_m": round(sum(l.get("climb_m", 0) for l in legs)), "drop_m": round(sum(l.get("drop_m", 0) for l in legs)),
            "earthwork_m3": round(sum(l["earthwork_m3"] for l in legs)),
            "grading_sols": round(grading, 1), "clearing_sols": round(clearing, 1), "bore_sols": round(bore_sols, 1),
            "build_sols": round(grading + clearing + bore_sols, 1),
            "score_sols": round(sc["score_sols"], 1), "shortest_score_sols": round(no_bore, 1),
            "beats_shortest": bool(sc["score_sols"] < no_bore),
            "basis": f"build sols + driving {max(ctx.fleet.trips_per_sol, 1)} round trip(s)/sol over {HORIZON_SOLS} sols, "
                     f"climbs and bore height weighted",
        }

    def line(self) -> dict:
        n = len(self.nodes)
        idx = np.unique(np.linspace(0, n - 1, min(n, LINE_ROWS)).round().astype(int))
        return {"from": list(self.A), "to": list(self.B), "length_km": self.km(n - 1),
                "lowest_bore_m": round(self.lo), "highest_ground_m": round(self.hi),
                "profile": [[self.km(k), round(float(self.H[k])), round(float(self.slope[k]), 1)] for k in idx]}

    def scan(self) -> list[tuple[float, dict]]:
        levels = np.linspace(self.lo, self.hi, SCAN_LEVELS, endpoint=False)
        return [(float(h), self.score(float(h))) for h in levels]

    def best(self) -> dict:
        """The rule: the bore height with the lowest score (a scan of heights), returned even when the shortest
        route scores better (flagged `beats_shortest: false`) so it can be weighed. No bore if none is buildable
        or none beats driving over the top along the line."""
        flat = self.score(None)
        cands = [(s["score_sols"], h) for h, s in self.scan() if s["ok"] and s["runs"]]
        if flat["ok"]:
            cands.append((flat["score_sols"], None))
        if not cands:
            out = self.evaluate(None)
            out.update(decided_by="rule", line=self.line(), rationale="No bore height gives a buildable road along the line.")
            return out
        _, h = min(cands, key=lambda c: c[0])
        plan = self.evaluate(h)
        plan.update(decided_by="rule", line=self.line(),
                    rationale=(f"Lowest score of {SCAN_LEVELS} heights from {self.lo:.0f} m to {self.hi:.0f} m: bore level "
                               f"at {h:.0f} m." if h is not None else "Driving over the top scores best: no bore."))
        plan["note"] = ("scores better than the shortest route" if plan.get("beats_shortest") else
                        "the shortest route scores better; shown so the bore can be weighed")
        return plan


def _leg_view(r: dict) -> dict:
    keep = ("geometry", "length_km", "travel_h", "climb_m", "drop_m", "max_grade_deg", "earthwork_m3", "grading_sols",
            "clearing_sols", "bore_sols", "steep", "design_profile", "segments")
    return {k: r[k] for k in keep if k in r}


# ---------- Claude decides the height ----------

SYSTEM = """You plan a road for a robotic construction fleet on Mars. The road runs along the straight line from A
to B; its route is fixed. Your only decision is the height at which a tunnel boring machine (TBM) bores: every
stretch where the ground rises above that height becomes a level bore at that height (vehicles drive through it),
and the rest is a built road on the line. A lower height means less climbing and descending on the road but
longer bores, which take boring time; a height at or above the highest ground means no bore. The aim is to avoid
long climbs and descents where boring pays for itself. Use the elevation profile you are given.
Try heights with evaluate_height (a few, comparing scores; lower is better), then call submit_height with your
choice and two or three sentences on why."""

TOOLS = [
    {"name": "evaluate_height",
     "description": "Cost a bore height (metres, same datum as the profile). null means no bore. Returns feasibility, "
                    "score_sols (lower is better), the bores it makes and the road's climb and earthwork.",
     "input_schema": {"type": "object", "properties": {"height_m": {"type": ["number", "null"]}},
                      "required": ["height_m"]}},
    {"name": "submit_height",
     "description": "Your chosen bore height (null for no bore) and why.",
     "input_schema": {"type": "object", "properties": {"height_m": {"type": ["number", "null"]},
                                                       "rationale": {"type": "string"}},
                      "required": ["height_m", "rationale"]}},
]
MAX_TURNS = 10


def _thin(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [rows[i] for i in idx]


def brief(th: Through) -> dict:
    req, fl = th.req, th.ctx.fleet
    return {
        "fleet": {"max_grade_deg": fl.limits.max_grade_deg, "road_width_m": fl.road_width_m,
                  "trips_per_sol": max(fl.trips_per_sol, 1), "slowest_kmh": fl.slowest_kmh},
        "tbm": {"name": th.raw_tbm.name, "bore_m_per_sol": round(th.rate, 1), "setup_sols_per_bore": BORE_SETUP_SOLS,
                "min_cover_m": th.tbm.min_cover_m, "max_tunnel_km": req.max_tunnel_km, "min_bore_m": MIN_BORE_M},
        "scoring": {"drive_sols_per_m": round(th.drive_m, 5), "tunnel_vertical_m": req.tunnel_vertical_m,
                    "resurface_penalty_sols": req.resurface_penalty_sols,
                    "basis": "score = build sols + driving over 500 sols, climbs weighted; each bore after the first adds "
                             "the resurface penalty"},
        "line": {"length_km": th.km(len(th.nodes) - 1), "ground_at_A_m": round(float(th.H[0])),
                 "ground_at_B_m": round(float(th.H[-1])), "lowest_bore_height_m": round(th.lo),
                 "highest_ground_m": round(th.hi),
                 "profile_km_ground_slopedeg": _thin(th.line()["profile"], 110)},
        "shortest_route_score_sols": round(th.shortest_m * th.drive_m, 1),
    }


def _compact(plan: dict) -> dict:
    if plan["status"] != "ok":
        return {"feasible": False, "reason": plan["reason"]}
    keep = ("score_sols", "build_sols", "bore_sols", "grading_sols", "earthwork_m3", "climb_m", "drop_m",
            "length_km", "tunnel_km", "travel_h")
    return {"feasible": True, **{k: plan[k] for k in keep},
            "bores": [{k: b[k] for k in ("from_km", "to_km", "length_km", "bore_sols")}
                      | {"max_cover_m": b["cover"]["max_cover_m"]} for b in plan["bores"]]}


def plan_with_claude(th: Through) -> dict:
    """Claude reads the elevation along the line, tries heights and submits one. Falls back to the rule if the
    call fails or the submitted height isn't buildable."""
    import anthropic
    from construct import ai
    num = lambda x: None if x is None else float(x)
    msgs = [{"role": "user", "content": "Choose the bore height for the A→B road.\n" + json.dumps(brief(th))}]
    tries = []
    try:
        for _ in range(MAX_TURNS):
            resp = ai._get_client().beta.messages.create(
                model=ai.MODEL, max_tokens=8000, system=SYSTEM, tools=TOOLS, messages=msgs,
                output_config={"effort": "medium"}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            if resp.stop_reason == "refusal":
                raise ValueError("refused")
            msgs.append({"role": "assistant", "content": resp.content})
            results, submitted = [], None
            for blk in resp.content:
                if blk.type != "tool_use":
                    continue
                if blk.name == "submit_height":
                    submitted = blk.input
                    break
                plan = th.evaluate(num(blk.input.get("height_m")))
                c = _compact(plan)
                tries.append({"height_m": plan.get("height_m"), "tunnel_km": c.get("tunnel_km"),
                              **{k: c.get(k) for k in ("feasible", "score_sols", "reason")}})
                results.append({"type": "tool_result", "tool_use_id": blk.id, "content": json.dumps(c)})
            if submitted is not None:
                plan = th.evaluate(num(submitted.get("height_m")))
                if plan["status"] != "ok":
                    raise ValueError(f"submitted height isn't buildable: {plan['reason']}")
                plan.update(decided_by=f"ai:{resp.model}", rationale=submitted.get("rationale", ""), tries=tries,
                            line=th.line())
                if not plan["bores"]:
                    plan["note"] = "Claude chose no bore: the road goes over the top along the line"
                return plan
            msgs.append({"role": "user", "content": results or "Call submit_height with your choice."})
        raise ValueError("no height submitted")
    except (anthropic.AnthropicError, ValueError, TypeError, KeyError) as e:
        plan = th.best()
        plan.update(decided_by="rule (AI unavailable)", tries=tries,
                    rationale=f"{plan.get('rationale', '')} [AI call failed: {type(e).__name__}: {str(e)[:160]}]")
        return plan


def plan(t, req, vehicles, tbm) -> dict:
    """POST /routes/through: set up the search like /routes, then cost the given height, or let Claude or the
    rule choose it."""
    from .routing import Ctx, _snap, _window, make_fleet
    fleet = make_fleet(vehicles, req)
    a_rc, b_rc = t.rc(*req.from_), t.rc(*req.to)
    ctx = Ctx(t, t.grid(*_window(t, a_rc, b_rc, req.search_margin_km * 1000)), fleet, req.hazard_weights)
    a, _ = _snap(ctx, ctx.node(*a_rc))
    b, _ = _snap(ctx, ctx.node(*b_rc))
    th = Through(ctx, a, b, req, tbm)
    if req.height_m is not None or req.no_bore:
        out = th.evaluate(None if req.no_bore else req.height_m)
        out.update(decided_by="given", line=th.line())
    elif req.decide == "ai":
        out = plan_with_claude(th)
    else:
        out = th.best()
    if out.get("status") == "ok" and fleet.trips_per_sol:
        out["haulers_needed"] = math.ceil(fleet.trips_per_sol * 2 * out["length_km"] / fleet.hauler.speed_kmh
                                          / req.operating_hours_per_sol)
    return out
