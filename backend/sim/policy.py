"""Response menus, the fixed-rule baseline, and Monte Carlo evaluation.

Evaluation never peeks at hidden truth. Each rollout clones the simulation, redraws
everything the planner cannot see (the cause of an open anomaly from its posterior,
true cutter wear, hard-rock bands ahead of each TBM from the map's tile odds, storm
length, cargo delays), applies an option, and plays the rest of the mission with the fixed rule.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

from .engine import CAUSES, Sim, tbm_posterior
from .model import CH_RESAMPLE, u01

FIX_FOR = {"hard_rock": None, "worn_cutters": "cutters", "bearing": "bearing", "none": None}


def _tbms_running(sim: Sim) -> list[str]:
    return [t.id for t in sim.tbms if t.status == "ok"]


def menu(sim: Sim, inc) -> list[dict]:
    """Ready-made options for an incident. Each is a label plus a list of actions."""
    k, m = inc.kind, inc.target
    if k == "tbm_anomaly":
        opts = [
            {"label": "Keep boring as is", "actions": [{"type": "continue"}]},
            {"label": "Slow to half speed", "actions": [{"type": "set_speed", "machine": m, "speed": 0.5}]},
            {"label": "Change cutters now", "actions": [{"type": "repair", "machine": m, "part": "cutters"}]},
            {"label": "Replace main bearing now", "actions": [{"type": "repair", "machine": m, "part": "bearing"}]},
            {"label": "Inspect (8 crew-h, 1 sol), then fix what it finds", "actions": [{"type": "inspect_then_fix", "machine": m}]},
        ]
        if sim.comms_ok:
            opts.append({"label": "Ask Earth engineers (2 sols), then fix what they find",
                         "actions": [{"type": "consult_then_fix", "machine": m}]})
        return [o for o in opts if _feasible(sim, o)]
    if k == "tbm_breakdown":
        part = "bearing" if "bearing" in inc.truth.get("cause", "") else "cutters"
        other = [t.id for t in sim.tbms if t.id != m and t.status in ("ok", "repairing", "down")]
        opts = [{"label": f"Repair with a {part} spare", "actions": [{"type": "repair", "machine": m, "part": part}]}]
        for o in other:
            opts.append({"label": f"Strip {o} for its {part}", "actions": [{"type": "cannibalize", "from": o, "to": m, "part": part}]})
        opts.append({"label": "Leave it down", "actions": [{"type": "continue"}]})
        return [o for o in opts if _feasible(sim, o)]
    if k == "wheel_damage" and m == "dozer":
        donors = [h.id for h in sim.haulers if h.status in ("ok", "down")]
        opts = [{"label": "Repair the track with a wheel spare", "actions": [{"type": "repair", "machine": m, "part": "wheel"}]}]
        if donors:
            opts.append({"label": f"Strip {donors[-1]} for running gear", "actions": [
                {"type": "cannibalize", "from": donors[-1], "to": m, "part": "wheel"}]})
        opts.append({"label": "Leave it down", "actions": [{"type": "continue"}]})
        return [o for o in opts if _feasible(sim, o)]
    if k == "wheel_damage":
        alt = next((r for r in sim.site.routes if r != sim.route), None)
        opts = [{"label": "Replace the wheel", "actions": [{"type": "repair", "machine": m, "part": "wheel"}]}]
        if alt:
            opts.append({"label": f"Replace the wheel and switch to the {alt} route",
                         "actions": [{"type": "repair", "machine": m, "part": "wheel"},
                                     {"type": "switch_route", "route": alt}]})
        opts.append({"label": "Leave it down", "actions": [{"type": "continue"}]})
        return [o for o in opts if _feasible(sim, o)]
    if k == "dust_storm":
        tb = _tbms_running(sim)
        hl = [h.id for h in sim.haulers if h.status == "ok"]
        opts = [{"label": "Keep everything running", "actions": [{"type": "continue"}]}]
        if tb:
            opts.append({"label": "Pause TBMs until the storm ends",
                         "actions": [{"type": "pause", "targets": tb, "until": "storm_end"}]})
        opts += [
            {"label": "Pause TBMs, haulers and dozer until the storm ends (plant runs on its ore stock)",
             "actions": [{"type": "pause", "targets": tb + hl + ["dozer"], "until": "storm_end"}]},
            {"label": "Throttle the plant to 50% and pause TBMs",
             "actions": [{"type": "throttle_plant", "level": 0.5}] + ([{"type": "pause", "targets": tb, "until": "storm_end"}] if tb else [])},
        ]
        return opts
    if k == "plant_trip":
        tb = _tbms_running(sim)
        hl = [h.id for h in sim.haulers if h.status == "ok"]
        return [
            {"label": "Restart now", "actions": [{"type": "restart_plant"}]},
            {"label": "Pause TBMs, haulers and dozer, then restart",
             "actions": [{"type": "pause", "targets": tb + hl + ["dozer"], "until": "storm_end" if sim.storm else "manual"},
                         {"type": "restart_plant"}]},
            {"label": "Throttle to 50%, pause TBMs and haulers, then restart",
             "actions": [{"type": "throttle_plant", "level": 0.5},
                         {"type": "pause", "targets": tb + hl, "until": "storm_end" if sim.storm else "manual"},
                         {"type": "restart_plant"}]},
            {"label": "Wait (leave it off for now)", "actions": [{"type": "continue"}]},
        ]
    if k == "cargo_slip":
        tb = _tbms_running(sim)
        return [
            {"label": "No change", "actions": [{"type": "continue"}]},
            {"label": "Run TBMs at half speed to save spares",
             "actions": [{"type": "set_speed", "machine": t, "speed": 0.5} for t in tb] or [{"type": "continue"}]},
        ]
    if k == "resupply":
        acts = _repair_down(sim)
        opts = [{"label": "Repair every down machine", "actions": acts}] if acts else []
        return opts + [{"label": "Keep the spares in reserve", "actions": [{"type": "continue"}]}]
    return [{"label": "Acknowledge", "actions": [{"type": "continue"}]}]


def _repair_down(sim: Sim) -> list[dict]:
    left = dict(sim.spares)
    acts = []
    for m in sim.machines:
        if m.status != "down":
            continue
        part = "wheel" if m.down_reason == "wheel" else "bearing" if "bearing" in m.down_reason else "cutters"
        if left.get(part, 0) > 0:
            left[part] -= 1
            acts.append({"type": "repair", "machine": m.id, "part": part})
    return acts


def _feasible(sim: Sim, opt: dict) -> bool:
    for a in opt["actions"]:
        if a["type"] == "repair" and sim.spares.get(a["part"], 0) < 1:
            return False
    return True


def apply(sim: Sim, inc_id: str | None, actions: list[dict]) -> list[dict]:
    """Like Sim.act, plus the two compound actions used by menus and rollouts."""
    inc = next((i for i in sim.incidents if i.id == inc_id), None) if inc_id else None
    target = lambda: inc_id if inc is not None and inc.status == "open" else None
    out, plain = [], []

    def flush():
        if plain:
            out.extend(sim.act(target(), list(plain)))
            plain.clear()

    for a in actions:
        if a["type"] not in ("inspect_then_fix", "consult_then_fix"):
            plain.append(a)
            continue
        flush()
        kind = "inspect" if a["type"] == "inspect_then_fix" else "consult_earth"
        r = sim.act(target(), [{"type": kind}])[0] if target() else {"ok": False, "error": "incident closed"}
        out.append(r)
        if not r.get("ok"):
            continue
        part = FIX_FOR[r["finding"]]
        m = a["machine"]
        if part and sim.spares.get(part, 0) > 0:
            follow = {"type": "repair", "machine": m, "part": part}
        elif part:
            follow = {"type": "set_speed", "machine": m, "speed": 0.5}
        else:
            follow = {"type": "continue"}
        out.extend(sim.act(target(), [follow]))
    flush()
    return out


def play(sim: Sim, policy) -> dict:
    """Run to the end, letting `policy(sim, incident) -> actions` answer each incident."""
    guard = 0
    while not sim.done and guard < 5000:
        for inc in sim.open_incidents:
            apply(sim, inc.id, policy(sim, inc) or [{"type": "continue"}])
            if inc.status == "open":           # only a diagnosis was asked for
                sim.act(inc.id, [{"type": "continue"}])
        sim.advance()
        guard += 1
    return sim.outcome()


def fixed_rule(sim: Sim, inc) -> list[dict]:
    """What a crew without a planner would do: fix the most common cause right away."""
    k, m = inc.kind, inc.target
    if k == "tbm_anomaly":
        recent = any(i.target == m and i.sol >= sim.sol - 30 and i.id != inc.id and
                     any(a.get("part") == "cutters" for a in i.actions) for i in sim.incidents)
        for part in (("bearing", "cutters") if recent else ("cutters", "bearing")):
            if sim.spares.get(part, 0) > 0:
                return [{"type": "repair", "machine": m, "part": part}]
        return [{"type": "set_speed", "machine": m, "speed": 0.5}]
    if k == "tbm_breakdown":
        part = "bearing" if "bearing" in inc.truth.get("cause", "") else "cutters"
        return [{"type": "repair", "machine": m, "part": part}] if sim.spares.get(part, 0) else [{"type": "continue"}]
    if k == "wheel_damage":
        return [{"type": "repair", "machine": m, "part": "wheel"}] if sim.spares.get("wheel", 0) else [{"type": "continue"}]
    if k == "plant_trip":
        return [{"type": "restart_plant"}]
    if k == "resupply":
        return _repair_down(sim) or [{"type": "continue"}]
    return [{"type": "continue"}]


# rollouts

def resample_hidden(c: Sim, k: int):
    """Replace hidden truth with a draw consistent with what has been observed."""
    p = c.p
    c.seed = k
    for i, t in enumerate(c.tbms):
        inc = next((x for x in c.incidents if x.status == "open" and x.kind == "tbm_anomaly" and x.target == t.id), None)
        cause = "none"
        if inc is not None:
            post = tbm_posterior(inc.symptoms, inc.symptoms["est_cutter_wear"])
            u, acc = u01(k, i, CH_RESAMPLE, 1), 0.0
            for cname in CAUSES:
                acc += post[cname]
                if u < acc:
                    cause = cname
                    break
            inc.truth = {"cause": cause}
        est = t.est_wear
        t.cutter_wear = max(0.0, est * (0.85 + 0.4 * u01(k, i, CH_RESAMPLE, 2)))
        t.hard_zones = c._draw_zones(k, i, t.position_m)
        t.bearing_failing, t.bearing_health = False, 1.0
        if t.status == "down" and t.down_reason == "bearing_seized":
            t.bearing_failing, t.bearing_health = True, 0.0
        if t.status == "down" and t.down_reason == "cutters_worn":
            t.cutter_wear = 1.0
        if cause == "bearing":
            t.bearing_failing, t.bearing_health = True, 0.25 + 0.3 * u01(k, i, CH_RESAMPLE, 3)
        elif cause == "worn_cutters":
            t.cutter_wear = 0.85 + 0.12 * u01(k, i, CH_RESAMPLE, 4)
        elif cause == "hard_rock":
            ln = 40 + 110 * u01(k, i, CH_RESAMPLE, 5)
            t.hard_zones.insert(0, (t.position_m - 1, t.position_m + ln))
    if c.storm:
        lo, hi = p.storm_len
        elapsed = c.sol - c.storm["start"]
        total = lo + u01(k, 0, CH_RESAMPLE, 6) * (hi - lo)
        c.storm["end"] = c.storm["start"] + max(int(total), elapsed + 1)
    if not c.cargo_landed and (c.cargo_notice is None or c.cargo_notice["sol"] > c.sol):
        c.cargo_arrival = p.resupply_sol
        c.cargo_notice = None
        lo, hi = p.cargo_notice_sols
        if u01(k, 0, CH_RESAMPLE, 7) < p.cargo_slip_prob:
            n_sol = int(lo + u01(k, 0, CH_RESAMPLE, 8) * (hi - lo))
            if n_sol > c.sol:
                slip = p.cargo_slip_sols[0] + u01(k, 0, CH_RESAMPLE, 9) * (p.cargo_slip_sols[1] - p.cargo_slip_sols[0])
                c.cargo_notice = {"sol": n_sol, "slip": int(slip)}


def _rollout_sim(sim: Sim, k: int) -> Sim:
    c = sim.clone()
    c.log = []
    c.series = None
    resample_hidden(c, k)
    return c


def _rollouts(sim: Sim, actions: list, incident_id: str | None, idx: range, continuation=None):
    continuation = continuation or fixed_rule
    finishes, spares = [], []
    for i in idx:
        k = int(u01(sim.seed, sim.sol, CH_RESAMPLE, 1000 + i) * 2**62)
        c = _rollout_sim(sim, k)
        if actions:
            apply(c, incident_id, actions)
        play(c, continuation)
        finishes.append(c.finish_sol if c.finish_sol is not None else math.inf)
        spares.append(sum(c.spares.values()))
    return finishes, spares


_POOL: ProcessPoolExecutor | None = None
PARALLEL_MIN = 120          # rollouts per call before it is worth farming out


def _pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        # fork, so workers never re-import the caller's script (forkserver would)
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        _POOL = ProcessPoolExecutor(max_workers=max(1, min(24, (os.cpu_count() or 2) - 1)), mp_context=ctx)
    return _POOL


def evaluate(sim: Sim, options: list[dict], incident_id: str | None = None, n: int = 60,
             continuation=fixed_rule, parallel: bool | None = None) -> list[dict]:
    """Chance of meeting the deadline under each option, by Monte Carlo rollouts.
    Rollout i uses the same random draws under every option (common random numbers)."""
    deadline = sim.p.deadline_sol
    use_pool = parallel if parallel is not None else (n * len(options) >= PARALLEL_MIN and continuation is fixed_rule)
    if use_pool:
        pool = _pool()
        chunk = max(4, n // 8)
        jobs = [[pool.submit(_rollouts, sim, opt.get("actions", []), incident_id, range(a, min(n, a + chunk)))
                 for a in range(0, n, chunk)] for opt in options]
        raw = []
        for fs in jobs:
            f_all, s_all = [], []
            for f in fs:
                a, b = f.result()
                f_all += a
                s_all += b
            raw.append((f_all, s_all))
    else:
        raw = [_rollouts(sim, opt.get("actions", []), incident_id, range(n), continuation) for opt in options]
    results = []
    for opt, (finishes, spares) in zip(options, raw):
        ok = [f for f in finishes if f <= deadline]
        finite = sorted(f for f in finishes if f < math.inf)
        results.append({
            "label": opt.get("label", ""),
            "actions": opt.get("actions", []),
            "p_on_time": round(len(ok) / n, 3),
            "median_finish_sol": finite[len(finite) // 2] if len(finite) > n / 2 else None,
            "p10_finish_sol": finite[int(0.1 * n)] if len(finite) > 0.1 * n else None,
            "p90_finish_sol": finite[int(0.9 * n)] if len(finite) > 0.9 * n else None,
            "mean_spares_left": round(statistics.mean(spares), 1),
            "runs": n,
        })
    return results


def forecast(sim: Sim, n: int = 80) -> dict:
    """Chance of making the window if the remaining plan runs with the fixed rule."""
    opens = sim.open_incidents
    r = evaluate(sim, [{"label": "as planned", "actions": []}], opens[0].id if opens else None, n=n)[0]
    return {k: r[k] for k in ("p_on_time", "median_finish_sol", "p10_finish_sol", "p90_finish_sol", "runs")}


TIE = 0.02    # options this close in odds count as tied


def recommend(results: list[dict]) -> dict:
    """Best odds of making the window. Near-ties go to the option that gathers
    information (inspect / ask Earth), then to the earlier median finish."""
    top = max(r["p_on_time"] for r in results)
    close = [r for r in results if r["p_on_time"] >= top - TIE]
    diag = lambda r: any(a["type"] in ("inspect_then_fix", "consult_then_fix") for a in r["actions"])
    return max(close, key=lambda r: (diag(r), -(r["median_finish_sol"] or 10**6), r["p_on_time"]))


def sim_guided(n: int = 16):
    """Policy that scores every menu option by rollouts and takes the best.
    Stands in for the AI planner in batch scorecards: same information, no hidden truth."""
    def policy(sim: Sim, inc) -> list[dict]:
        opts = menu(sim, inc)
        if len(opts) == 1:
            return opts[0]["actions"]
        return recommend(evaluate(sim, opts, inc.id, n=n, parallel=False))["actions"]
    return policy
