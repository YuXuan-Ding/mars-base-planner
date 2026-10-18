"""Sol-by-sol construction simulator with hidden faults.

The simulator keeps the true state (bearing health, real cutter wear, hard-rock zones,
future weather). Incidents come from the odds on the map tiles where machines work
(hazmap.py) and from seasonal weather and cargo odds; nothing is scripted.
Callers only ever get `observe()`: telemetry, reports and incidents
with symptoms. The true cause of an incident is revealed after it is resolved.
"""
from __future__ import annotations

import copy
import math
import zlib

from .hazmap import site_for
from .model import (CH_BEARING, CH_CARGO, CH_FINDING, CH_NOISE, CH_NOTE, CH_STORM, CH_STORM_F,
                    CH_STORM_LEN, CH_WHEEL, CH_ZONES, TBM, Dozer, Hauler, Incident, Params, Plant, gauss, u01)

CAUSES = ("hard_rock", "worn_cutters", "bearing", "none")
NOTES = {
    "grinding": "Grinding noise from the main drive since last shift.",
    "harder_face": "Face looks darker and harder than the logged unit.",
    "chipped": "Two disc cutters looked chipped at the last check.",
    "nothing": "Nothing unusual seen at the face.",
}
# P(crew note | cause)
NOTE_P = {
    "hard_rock": {"grinding": .10, "harder_face": .60, "chipped": .15, "nothing": .15},
    "worn_cutters": {"grinding": .15, "harder_face": .10, "chipped": .55, "nothing": .20},
    "bearing": {"grinding": .60, "harder_face": .10, "chipped": .10, "nothing": .20},
    "none": {"grinding": .10, "harder_face": .10, "chipped": .10, "nothing": .70},
}
FINDING_TEXT = {
    "hard_rock": "Face is in a hard basaltic band; cutters and drive are fine.",
    "worn_cutters": "Most disc cutters are near their wear limit; several chipped.",
    "bearing": "Main bearing race is spalling; failure likely within about ten sols.",
    "none": "No fault found; readings look like sensor noise.",
}
REF_HARD_PER_KM = Params.hard_bands_per_km
SPARE_FOR = {"cutters": "cutters", "bearing": "bearing", "wheel": "wheel"}


def seed_int(seed: int | str) -> int:
    return seed if isinstance(seed, int) else zlib.crc32(str(seed).encode())


def symptom_means(hard: float, wear: float, sev: float) -> tuple[float, float, float]:
    """Expected (torque ratio, vibration mm/s, advance ratio) for a TBM state."""
    torque = 1 + 0.55 * hard + 1.2 * max(0.0, wear - 0.65) + 0.3 * sev
    vib = 3.0 + 5.5 * sev + 0.9 * hard + 2.0 * max(0.0, wear - 0.7)
    adv = (0.5 if hard else 1.0) * wear_factor(wear)
    return torque, vib, adv


def wear_factor(wear: float) -> float:
    return 1.0 if wear < 0.8 else max(0.3, 1 - (wear - 0.8) * 3.5)


def tbm_posterior(sym: dict, est_wear: float) -> dict[str, float]:
    """P(cause | symptoms), using only what the crew can observe."""
    w = min(max((est_wear - 0.5) / 0.4, 0.0), 1.0)
    # the map's hard-rock odds for this tile shift the prior (0.35 on an average tile)
    hard = 0.35 * min(max(sym.get("tile_hard_per_km", REF_HARD_PER_KM) / REF_HARD_PER_KM, 0.4), 2.2)
    prior = {"hard_rock": hard, "worn_cutters": 0.15 + 0.6 * w, "bearing": 0.25, "none": 0.1}
    # (weight, hard, wear, bearing severity): a fault can sit inside a hard band too
    configs = {
        "hard_rock": [(1.0, 1.0, est_wear, 0.0)],
        "worn_cutters": [(0.35, h, wr, 0.0) for h in (0.0,) for wr in (0.85, 0.95)]
                        + [(0.15, 1.0, wr, 0.0) for wr in (0.85, 0.95)],
        "bearing": [(0.175, 0.0, est_wear, sv) for sv in (0.3, 0.45, 0.6, 0.75)]
                   + [(0.075, 1.0, est_wear, sv) for sv in (0.3, 0.45, 0.6, 0.75)],
        "none": [(1.0, 0.0, est_wear, 0.0)],
    }
    sd = (0.1, 0.8, 0.1)
    keys = ("torque_ratio", "vibration_mm_s", "advance_ratio")
    post = {}
    for c in CAUSES:
        lik = 0.0
        for wt, h, wr, sv in configs[c]:
            m = symptom_means(h, wr, sv)
            lik += wt * math.exp(sum(-0.5 * ((sym[k] - mu) / s) ** 2 for k, mu, s in zip(keys, m, sd)))
        for note in sym.get("notes", []):
            lik *= NOTE_P[c][note]
        post[c] = prior[c] * lik
    tot = sum(post.values()) or 1.0
    return {c: v / tot for c, v in post.items()}


class Sim:
    def __init__(self, params: Params | None = None, seed: int | str = 1):
        self.p = p = params or Params()
        self.site = site_for(p)          # shared by clones; never changes during a run
        self.seed = seed_int(seed)
        self.sol = 0
        self.tbms = [TBM(i, t, self.site.tunnels[t]["access_m"], p.cutter_wear_per_m) for i, t in p.tbms]
        self.haulers = [Hauler(h) for h in p.haulers]
        self.dozer = Dozer()
        self.plant = Plant()
        self.spares = dict(p.spares)
        if p.route not in self.site.routes:
            raise ValueError(f"route '{p.route}' not available here (have {list(self.site.routes)})")
        self.route = p.route
        self.solar_kw = 0.0
        self.dust = 0.0
        self.storm: dict | None = None
        self.spoil = 0.0
        self.pad_done = 0.0
        self.ore_portal = 0.0
        self.ore_plant = 0.0
        self.propellant = 0.0
        self.crew_debt = 0.0
        self.finish_sol: int | None = None
        self.cargo_arrival = p.resupply_sol
        self.cargo_landed = False
        self.cargo_notice: dict | None = None
        self.incidents: list[Incident] = []
        self.log: list[tuple[int, str]] = []
        self.series: dict[str, list] = {k: [] for k in ("sol", "propellant_t", "power_frac", "pad_pct", "solar_kw", "ore_plant_t")}
        self.last = {"supply_kw": 0.0, "demand_kw": 0.0, "power_frac": 1.0}
        self._events_sol = -1
        self._init_hidden()

    # setup

    def _init_hidden(self):
        p, s = self.p, self.seed
        for i, t in enumerate(self.tbms):
            t.hard_zones = self._draw_zones(s, i, 0.0)
        lo, hi = p.cargo_notice_sols
        if u01(s, 0, CH_CARGO, 1) < p.cargo_slip_prob:
            slip = p.cargo_slip_sols[0] + u01(s, 0, CH_CARGO, 2) * (p.cargo_slip_sols[1] - p.cargo_slip_sols[0])
            self.cargo_notice = {"sol": int(lo + u01(s, 0, CH_CARGO, 3) * (hi - lo)), "slip": int(slip)}

    def _draw_zones(self, s: int, i: int, after_m: float) -> list:
        """Hard-rock bands ahead of a TBM. Each tile under the tunnel has its own rate
        (hard_per_km); bands are drawn at the highest rate and kept in proportion
        to the rate of the tile they start in."""
        p, t = self.p, self.tbms[i]
        lam = self.site.max_hard_per_km(t.tunnel)
        zones, x, k = [], after_m, 0
        while x < after_m + 4000:
            k += 1
            x += -math.log(max(u01(s, i, CH_ZONES, k), 1e-9)) * 1000 / lam
            if u01(s, i, CH_ZONES, k, 3) * lam >= self.site.tunnel_tile(t.tunnel, x)["hard_per_km"]:
                continue
            ln = p.hard_zone_len[0] + u01(s, i, CH_ZONES, k, 7) * (p.hard_zone_len[1] - p.hard_zone_len[0])
            zones.append((x, x + ln))
            x += ln
        return zones

    # lookups

    @property
    def machines(self) -> list:
        return self.tbms + self.haulers + [self.dozer]

    def _tbm(self, mid: str) -> TBM:
        return next(t for t in self.tbms if t.id == mid)

    def machine(self, mid: str):
        for m in self.machines:
            if m.id == mid:
                return m
        raise KeyError(f"unknown machine {mid}")

    def _where(self, tile: dict, lonlat: list | None = None, **extra) -> dict:
        lon, lat = lonlat or (tile["lon"], tile["lat"])
        return {"lon": round(lon, 5), "lat": round(lat, 5), "tile": [tile["r"], tile["c"]],
                "tile_odds": {k: tile[k] for k in ("wheel_per_km", "hard_per_km", "bearing_mult", "dozer_per_1000m3")},
                **extra}

    @property
    def comms_ok(self) -> bool:
        a, b = self.p.conjunction
        return not (a <= self.sol <= b)

    @property
    def done(self) -> bool:
        return self.finish_sol is not None or self.sol > self.p.deadline_sol

    @property
    def open_incidents(self) -> list[Incident]:
        return [i for i in self.incidents if i.status == "open"]

    def _say(self, text: str):
        self.log.append((self.sol, text))

    def _raise(self, kind, target, title, report, symptoms=None, truth=None, where=None) -> Incident:
        inc = Incident(f"I{len(self.incidents) + 1}", kind, self.sol, target, title, report,
                       symptoms or {}, truth=truth or {}, where=where)
        self.incidents.append(inc)
        self._say(f"INCIDENT {inc.id}: {title}")
        return inc

    # world events

    def _events(self):
        p, s, sol = self.p, self.seed, self.sol
        if self.storm and sol >= self.storm["end"]:
            self._say("Dust storm over; sunlight back to normal.")
            self.storm = None
            for m in self.machines:
                if m.paused == "storm_end":
                    m.paused = None
            if self.plant.throttle_until == "storm_end":
                self.plant.throttle, self.plant.throttle_until = 1.0, None
        if not self.storm and p.storm_season[0] <= sol <= p.storm_season[1]:
            if u01(s, sol, CH_STORM) < p.storm_onset_per_sol:
                ln = p.storm_len[0] + u01(s, sol, CH_STORM_LEN) * (p.storm_len[1] - p.storm_len[0])
                f = p.storm_factor[0] + u01(s, sol, CH_STORM_F) * (p.storm_factor[1] - p.storm_factor[0])
                self._start_storm(int(ln), f)
        if self.cargo_notice and self.cargo_notice["sol"] == sol and not self.cargo_landed:
            self.cargo_arrival += self.cargo_notice["slip"]
            self._raise("cargo_slip", None, f"Resupply cargo slips {self.cargo_notice['slip']} sols",
                        f"Earth: resupply flight delayed. New landing estimate sol {self.cargo_arrival}. "
                        f"Spares on hand: {self._spares_text()}.")
        if not self.cargo_landed and sol >= self.cargo_arrival:
            self.cargo_landed = True
            for k, v in p.resupply.items():
                self.spares[k] = self.spares.get(k, 0) + v
            self._say(f"Resupply landed: {self._spares_text(p.resupply)} added.")
            down = [m.id for m in self.machines if m.status == "down"]
            if down:
                self._raise("resupply", None, "Resupply landed; machines waiting for parts",
                            f"Spares now: {self._spares_text()}. Down: {', '.join(down)}.")

    def _start_storm(self, length: int, factor: float):
        self.storm = {"start": self.sol, "end": self.sol + length, "factor": factor}
        self.dust = min(self.p.dust_max, self.dust + self.p.storm_dust)
        self._raise("dust_storm", None, "Regional dust storm",
                    f"Sky opacity rising fast. Solar output down to about {factor:.0%} of normal. "
                    f"Duration unknown (storms here last {self.p.storm_len[0]}-{self.p.storm_len[1]} sols)."
                    + ("" if self.comms_ok else " Earth is out of contact (solar conjunction)."),
                    truth={"length_sols": length, "factor": round(factor, 2)})

    def _spares_text(self, d=None) -> str:
        d = self.spares if d is None else d
        return ", ".join(f"{k} {v}" for k, v in d.items())

    # one sol

    def step(self):
        p, s, sol = self.p, self.seed, self.sol
        if self._events_sol != sol:
            self._events_sol = sol
            self._events()
            if self.open_incidents:      # let the planner respond before this sol's work
                return
        self._finish_repairs()
        for t in self.tbms:
            if t.status == "ok" and t.est_wear >= p.planned_change_wear and self.spares["cutters"] > 0 \
                    and not any(i.target == t.id and i.status == "open" for i in self.incidents):
                self.spares["cutters"] -= 1
                self._start_repair(t, "cutters", *p.cutter_change)
                self._say(f"{t.id}: routine cutter change ({self.spares['cutters']} sets left).")

        storm_f = self.storm["factor"] if self.storm else 1.0
        supply = p.lander_power_kw + self.solar_kw * (1 - self.dust) * storm_f
        ore_left = max(0.0, p.target_propellant_t - self.propellant) * p.ore_t_per_t
        enough_ore = self.ore_portal + self.ore_plant >= ore_left + p.ore_margin_t
        run_tbm = [t for t in self.tbms if t.status == "ok" and not t.paused
                   and not (enough_ore and t.position_m >= t.access_m)]
        run_haul = [h for h in self.haulers if h.status == "ok" and not h.paused] if self.ore_portal > 0 else []
        dz = self.dozer
        dozer_on = self.pad_done < p.pad_fill_m3 and self.spoil > 1 and dz.status == "ok" and not dz.paused
        pl = self.plant
        building = self.solar_kw < p.solar_target_kw or pl.status == "building"
        d_plant = p.plant_kw_per_t * p.plant_rate_t * pl.throttle * (
            1.0 if pl.status == "operating" else 0.5 if pl.status in ("commissioning", "restarting") else 0.0)
        demand = (sum(p.tbm_power_kw * t.speed for t in run_tbm) + p.hauler_power_kw * len(run_haul)
                  + (p.dozer_power_kw if dozer_on else 0) + d_plant + (p.construction_kw if building else 0))
        avail = max(0.0, supply - p.base_load_kw)
        frac = 1.0 if demand <= avail else avail / demand
        self.last = {"supply_kw": round(supply), "demand_kw": round(demand + p.base_load_kw), "power_frac": round(frac, 3)}

        # crew-hours: repairs already took theirs (crew_debt); construction shares the rest
        pool = p.crew_hours_per_sol - self.crew_debt
        self.crew_debt = max(0.0, -pool)
        pool = max(0.0, pool)
        need_solar = p.solar_crew_h_per_sol if self.solar_kw < p.solar_target_kw else 0.0
        need_plant = p.plant_crew_h_per_sol if pl.status == "building" else 0.0
        need = need_solar + need_plant
        share = min(1.0, pool / need) if need else 0.0

        # TBMs
        for i, t in enumerate(run_tbm):
            hard = t.in_hard()
            planned = p.tbm_rate_m * t.speed * frac
            rate = planned * (p.hard_rate_mult if hard else 1.0) * wear_factor(t.cutter_wear)
            before = t.position_m
            t.position_m += rate
            t.m_since_change += rate
            t.cutter_wear += rate * p.cutter_wear_per_m * (p.hard_wear_mult if hard else 1.0)
            self.spoil += rate * p.spoil_m3_per_m
            mined = max(0.0, t.position_m - max(t.access_m, before))
            self.ore_portal += mined * p.ore_t_per_m
            if before < t.access_m <= t.position_m:
                self._say(f"{t.id} finished the {t.tunnel} access tunnel; mining ore.")
            idx = self.tbms.index(t)
            if not t.bearing_failing:
                mult = self.site.tunnel_tile(t.tunnel, t.position_m)["bearing_mult"]
                if u01(s, sol, CH_BEARING, idx) < p.bearing_onset_per_sol * mult * t.speed:
                    t.bearing_failing, t.bearing_health = True, 1.0
            elif t.bearing_failing and frac > 0:
                t.bearing_health -= p.bearing_decay_per_sol * (p.reduced_decay_mult if t.speed < 1 else 1.0)
            self._tbm_telemetry(t, idx, hard, rate, planned)
            if t.bearing_failing and t.bearing_health <= 0:
                self._breakdown(t, "bearing_seized", "Main bearing seized; TBM stopped.")
            elif t.cutter_wear >= 1.0:
                self._breakdown(t, "cutters_worn", "Cutters worn out; TBM cannot advance.")
            else:
                self._check_anomaly(t, frac)

        # haulers
        r = self.site.routes[self.route]
        trips = p.hauler_ops_h / (2 * r["km"] / p.hauler_speed_kmh + p.hauler_load_h)
        cap = trips * p.hauler_payload_t * frac
        moved = 0.0
        want = max(0.0, min(self.ore_portal, p.plant_buffer_t - self.ore_plant))
        for n_left, h in zip(range(len(run_haul), 0, -1), run_haul):
            take = min(cap, (want - moved) / n_left)
            if take <= 0:
                break
            moved += take
            dist = take / p.hauler_payload_t * 2 * r["km"]
            h.km += dist
            hi = self.haulers.index(h)
            # every round trip crosses each tile of the route twice; odds add up tile by tile
            pdmg = 1 - math.exp(-r["hits_per_trip"] * dist / (2 * r["km"]))
            if u01(s, sol, CH_WHEEL, hi) < pdmg:
                tile = self._pick_tile(r["segs"], u01(s, sol, CH_WHEEL, hi, 5))
                h.status, h.down_reason = "down", "wheel"
                self._raise("wheel_damage", h.id, f"{h.id} wheel damaged on the {self.route} route",
                            f"{h.id} cracked a wheel on a rock near {tile['lon']:.3f}E {tile['lat']:.3f}N "
                            f"({tile['rock_cover_025_pct']:.1f}% of that tile is covered by rocks over 25 cm). "
                            f"Hauler stopped. Wheel spares: {self.spares['wheel']}.",
                            where=self._where(tile, route=self.route))
        self.ore_portal -= moved
        self.ore_plant += moved

        # pad
        if dozer_on:
            tile = self.site.pad_tile(self.pad_done / p.pad_fill_m3)
            g = min(p.dozer_m3_per_sol * frac, self.spoil, p.pad_fill_m3 - self.pad_done)
            self.pad_done += g
            self.spoil -= g
            if self.pad_done >= p.pad_fill_m3:
                self._say("Launch pad graded and finished.")
            elif u01(s, sol, CH_WHEEL, 99) < 1 - math.exp(-tile["dozer_per_1000m3"] * g / 1000):
                dz.status, dz.down_reason = "down", "wheel"
                self._raise("wheel_damage", dz.id, "Dozer track damaged on a buried boulder",
                            f"The dozer hit a buried boulder grading the pad near {tile['lon']:.3f}E {tile['lat']:.3f}N "
                            f"and broke a track roller. Pad {100 * self.pad_done / p.pad_fill_m3:.0f}% done. "
                            f"Running-gear (wheel) spares: {self.spares['wheel']}.",
                            where=self._where(tile, job="pad"))

        # solar
        if need_solar:
            self.solar_kw = min(p.solar_target_kw, self.solar_kw + need_solar * share * p.solar_kw_per_crew_h * frac)
            if self.solar_kw >= p.solar_target_kw:
                self._say("Solar field complete.")

        # plant
        if pl.status == "building":
            pl.build_h += need_plant * share * frac
            if pl.build_h >= p.plant_build_crew_h:
                pl.status, pl.commission_left = "commissioning", p.commission_sols
                self._say("Propellant plant built; commissioning.")
        elif pl.status == "commissioning":
            if frac >= p.plant_trip_frac:
                pl.commission_left -= 1
                if pl.commission_left <= 0:
                    pl.status = "operating"
                    self._say("Propellant plant operating.")
        elif pl.status == "restarting" and sol >= pl.restart_until:
            pl.status = "operating" if frac >= p.plant_trip_frac else "tripped"
            if pl.status == "tripped":
                self._trip("Plant restart failed: not enough power.")
            else:
                self._say("Plant restarted.")
        elif pl.status == "operating":
            if frac < p.plant_trip_frac:
                self._trip(f"Plant tripped offline: it got {frac:.0%} of its power (needs {p.plant_trip_frac:.0%}).")
            else:
                make = min(p.plant_rate_t * pl.throttle * frac, self.ore_plant / p.ore_t_per_t)
                self.propellant += make
                self.ore_plant -= make * p.ore_t_per_t

        self.dust = min(p.dust_max, self.dust + p.dust_per_sol)
        if self.finish_sol is None and self.propellant >= p.target_propellant_t and self.pad_done >= p.pad_fill_m3:
            self.finish_sol = sol
            self._say("GOAL MET: propellant loaded and pad ready.")
        ser = self.series
        if ser is None:
            self.sol += 1
            return
        ser["sol"].append(sol)
        ser["propellant_t"].append(round(self.propellant, 1))
        ser["power_frac"].append(round(frac, 3))
        ser["pad_pct"].append(round(100 * self.pad_done / p.pad_fill_m3, 1))
        ser["solar_kw"].append(round(self.solar_kw))
        ser["ore_plant_t"].append(round(self.ore_plant))
        self.sol += 1

    def _pick_tile(self, segs: list, u: float) -> dict:
        """A tile along a route, chosen in proportion to its share of the damage odds."""
        w = [d * self.site.tiles[ti]["wheel_per_km"] for d, ti in segs]
        acc, target = 0.0, u * sum(w)
        for wi, (_, ti) in zip(w, segs):
            acc += wi
            if acc >= target:
                return self.site.tiles[ti]
        return self.site.tiles[segs[-1][1]]

    def _trip(self, msg: str):
        self.plant.status = "tripped"
        self._raise("plant_trip", "plant", "Propellant plant tripped offline",
                    f"{msg} Restart takes {self.p.plant_restart[1]} sols and {self.p.plant_restart[0]:.0f} crew-hours, "
                    f"and needs at least {self.p.plant_trip_frac:.0%} power when it comes back.")

    def _breakdown(self, t: TBM, reason: str, text: str):
        t.status, t.down_reason = "down", reason
        part = "bearing" if reason == "bearing_seized" else "cutters"
        self._raise("tbm_breakdown", t.id, f"{t.id} stopped: {reason.replace('_', ' ')}",
                    f"{text} Needs a {part} spare (on hand: {self.spares[part]}).", truth={"cause": reason},
                    where=self._tbm_where(t))

    def _tbm_telemetry(self, t: TBM, idx: int, hard: bool, rate: float, planned: float):
        sev = (1 - t.bearing_health) if t.bearing_failing else 0.0
        tq, vib, _ = symptom_means(1.0 if hard else 0.0, t.cutter_wear, sev)
        s, sol = self.seed, self.sol
        t.telemetry = {
            "torque_ratio": round(tq + 0.08 * gauss(s, sol, CH_NOISE, idx, 1), 2),
            "vibration_mm_s": round(max(0.5, vib + 0.6 * gauss(s, sol, CH_NOISE, idx, 2)), 1),
            "advance_ratio": round(min(1.2, rate / planned + 0.05 * gauss(s, sol, CH_NOISE, idx, 3)), 2) if planned else 0.0,
            "advance_m": round(rate, 2),
        }

    def _true_cause(self, t: TBM) -> str:
        if t.bearing_failing and t.bearing_health < 0.75:
            return "bearing"
        if t.in_hard():
            return "hard_rock"
        if t.cutter_wear > 0.8:
            return "worn_cutters"
        return "none"

    def _check_anomaly(self, t: TBM, frac: float):
        tel = t.telemetry
        if frac < 0.85 or self.sol - t.last_anomaly_sol <= self.p.anomaly_cooldown:
            return
        if any(i.target == t.id and i.status == "open" for i in self.incidents):
            return
        ack = t.ack or {"torque_ratio": 1.2, "vibration_mm_s": 5.0, "advance_ratio": 0.8}
        normal = tel["torque_ratio"] <= 1.35 and tel["vibration_mm_s"] <= 6.0 and tel["advance_ratio"] >= 0.65
        if normal:
            t.ack = None
            return
        worse = (tel["torque_ratio"] > max(1.35, ack["torque_ratio"] + 0.15)
                 or tel["vibration_mm_s"] > max(6.0, ack["vibration_mm_s"] + 1.0)
                 or tel["advance_ratio"] < min(0.65, ack["advance_ratio"] - 0.15))
        if not worse:
            return
        t.last_anomaly_sol = self.sol
        cause = self._true_cause(t)
        u, acc = u01(self.seed, self.sol, CH_NOTE, self.tbms.index(t)), 0.0
        notes = ["nothing"]
        for n, pr in NOTE_P[cause].items():
            acc += pr
            if u < acc:
                notes = [n]
                break
        tile = self.site.tunnel_tile(t.tunnel, t.position_m)
        sym = {**{k: tel[k] for k in ("torque_ratio", "vibration_mm_s", "advance_ratio")},
               "notes": notes, "est_cutter_wear": round(t.est_wear, 2), "position_m": round(t.position_m),
               "tile_hard_per_km": tile["hard_per_km"]}
        place = f"{t.tunnel} at {t.position_m:.0f} m" + (" (ore drifts)" if t.position_m > t.access_m else "")
        report = (f"{t.id} · {place}: cutter torque {tel['torque_ratio']:.2f}x normal, "
                  f"vibration {tel['vibration_mm_s']} mm/s (limit 6.0), advance {tel['advance_ratio']:.0%} of plan. "
                  f"Estimated cutter wear {t.est_wear:.0%}. The map expects {tile['hard_per_km']:.2f} hard bands/km "
                  f"under this tile. Crew log: \"{' '.join(NOTES[n] for n in notes)}\"")
        self._raise("tbm_anomaly", t.id, f"{t.id} anomaly", report, sym,
                    truth={"cause": cause, "bearing_health": round(t.bearing_health, 2) if t.bearing_failing else 1.0,
                           "cutter_wear": round(t.cutter_wear, 2), "in_hard_rock": t.in_hard()},
                    where=self._tbm_where(t))

    def _tbm_where(self, t: TBM) -> dict:
        tile = self.site.tunnel_tile(t.tunnel, t.position_m)
        return self._where(tile, self.site.tunnel_point(t.tunnel, t.position_m), tunnel=t.tunnel,
                           position_m=round(t.position_m))

    def _finish_repairs(self):
        p = self.p
        for m in self.machines:
            if m.status != "repairing" or self.sol < m.repair_until:
                continue
            fix = m.repair_fix
            if isinstance(m, TBM):
                if fix == "cutters":
                    m.cutter_wear = m.m_since_change = 0.0
                elif fix in ("bearing", "bearing_seized"):
                    m.bearing_failing, m.bearing_health = False, 1.0
            m.status, m.repair_fix, m.down_reason = "ok", None, ""
            self._say(f"{m.id} back in service" + (f" ({fix} done)" if fix and fix not in ("inspect", "consult") else "") + ".")

    # actions

    def act(self, incident_id: str | None, actions: list[dict]) -> list[dict]:
        """Apply actions. Any action other than a diagnosis resolves the incident."""
        inc = None
        if incident_id:
            inc = next((i for i in self.incidents if i.id == incident_id), None)
            if inc is None:
                raise ValueError(f"unknown incident {incident_id}")
            if inc.status != "open":
                raise ValueError(f"incident {incident_id} is already resolved")
        out = []
        resolves = False
        for a in actions:
            res = self._apply(a, inc)
            out.append(res)
            if res.get("ok") and a["type"] not in ("inspect", "consult_earth"):
                resolves = True
        if inc is not None:
            inc.actions.extend(actions)
            if resolves or any(a["type"] == "continue" for a in actions):
                inc.status, inc.resolved_sol = "resolved", self.sol
                if inc.kind == "tbm_anomaly":
                    t = self._tbm(inc.target)
                    repaired = any(x["type"] in ("repair", "cannibalize") for x in inc.actions)
                    t.ack = None if repaired else {k: inc.symptoms[k] for k in ("torque_ratio", "vibration_mm_s", "advance_ratio")}
        return out

    def _start_repair(self, m, fix: str, crew_h: float, sols: int):
        start = max(self.sol, m.repair_until) if m.status == "repairing" else self.sol
        m.status, m.repair_fix, m.repair_until = "repairing", fix, start + sols
        self.crew_debt += crew_h

    def _finding(self, inc: Incident, accuracy: float, tag: int) -> str:
        cause = inc.truth.get("cause", "none")
        if u01(self.seed, self.sol, CH_FINDING, int(inc.id[1:]), tag) < accuracy:
            return cause
        others = [c for c in CAUSES if c != cause]
        return others[int(u01(self.seed, self.sol, CH_FINDING, int(inc.id[1:]), tag, 9) * len(others))]

    def _apply(self, a: dict, inc: Incident | None) -> dict:
        p, t = self.p, a["type"]
        try:
            if t == "continue":
                return {"ok": True}
            if t in ("inspect", "consult_earth"):
                if inc is None or inc.kind != "tbm_anomaly":
                    return {"ok": False, "error": f"{t} applies to a TBM anomaly"}
                m = self.machine(inc.target)
                if t == "consult_earth":
                    if not self.comms_ok:
                        return {"ok": False, "error": "Earth is out of contact (solar conjunction)"}
                    self._start_repair(m, "consult", 0.0, p.consult_sols)
                    c = self._finding(inc, p.consult_accuracy, 2)
                else:
                    self._start_repair(m, "inspect", *p.inspect)
                    c = self._finding(inc, p.inspect_accuracy, 1)
                inc.findings.append({"by": t, "cause": c, "text": FINDING_TEXT[c]})
                return {"ok": True, "finding": c, "text": FINDING_TEXT[c]}
            if t == "repair":
                m, part = self.machine(a["machine"]), a["part"]
                if m.status == "stripped":
                    return {"ok": False, "error": f"{m.id} was stripped for parts"}
                if self.spares.get(part, 0) < 1:
                    return {"ok": False, "error": f"no {part} spare on hand"}
                if part == "wheel" and not isinstance(m, Hauler) or part != "wheel" and not isinstance(m, TBM):
                    return {"ok": False, "error": f"{part} does not fit {m.id}"}
                self.spares[part] -= 1
                if part == "bearing" and isinstance(m, TBM) and m.down_reason == "bearing_seized":
                    self._start_repair(m, "bearing_seized", *p.bearing_seized)
                else:
                    wheel = p.dozer_repair if isinstance(m, Dozer) else p.wheel_repair
                    self._start_repair(m, part, *{"cutters": p.cutter_change, "bearing": p.bearing_replace,
                                                  "wheel": wheel}[part])
                return {"ok": True, "back_in_service_sol": m.repair_until}
            if t == "cannibalize":
                src, dst, part = self.machine(a["from"]), self.machine(a["to"]), a["part"]
                if isinstance(src, TBM) != isinstance(dst, TBM) or src is dst or src.status == "stripped":
                    return {"ok": False, "error": "can only take parts from a machine of the same kind "
                                                  "(TBM to TBM, or hauler/dozer running gear)"}
                src.status, src.down_reason = "stripped", f"parts taken for {dst.id}"
                fix = "bearing_seized" if part == "bearing" and getattr(dst, "down_reason", "") == "bearing_seized" else part
                cost = {"cutters": p.cutter_change, "bearing": p.bearing_replace, "bearing_seized": p.bearing_seized,
                        "wheel": p.wheel_repair}[fix]
                self._start_repair(dst, fix, cost[0] * 1.5, cost[1] + 1)
                return {"ok": True, "back_in_service_sol": dst.repair_until}
            if t == "set_speed":
                m = self.machine(a["machine"])
                m.speed = float(a["speed"])
                return {"ok": True}
            if t in ("pause", "resume"):
                targets = a["targets"] if "targets" in a else [a["target"]]
                until = a.get("until", "manual")
                for tg in targets:
                    self.machine(tg).paused = None if t == "resume" else until
                return {"ok": True}
            if t == "restart_plant":
                if self.plant.status != "tripped":
                    return {"ok": False, "error": "plant is not tripped"}
                self.plant.status, self.plant.restart_until = "restarting", self.sol + p.plant_restart[1]
                self.crew_debt += p.plant_restart[0]
                return {"ok": True}
            if t == "throttle_plant":
                self.plant.throttle = min(1.0, max(0.2, float(a["level"])))
                self.plant.throttle_until = a.get("until", "storm_end" if self.storm else "manual")
                return {"ok": True}
            if t == "clean_panels":
                self.dust = 0.0
                self.crew_debt += p.clean_crew_h
                return {"ok": True}
            if t == "switch_route":
                if a["route"] not in self.site.routes:
                    return {"ok": False, "error": f"unknown route {a['route']}"}
                self.route = a["route"]
                return {"ok": True}
            return {"ok": False, "error": f"unknown action {t}"}
        except KeyError as e:
            return {"ok": False, "error": str(e)}

    # views

    def outcome(self) -> dict:
        return {
            "done": self.done,
            "finish_sol": self.finish_sol,
            "on_time": self.finish_sol is not None and self.finish_sol <= self.p.deadline_sol,
            "propellant_t": round(self.propellant, 1),
            "pad_pct": round(100 * self.pad_done / self.p.pad_fill_m3, 1),
        }

    def incident_view(self, inc: Incident) -> dict:
        d = {"id": inc.id, "kind": inc.kind, "sol": inc.sol, "target": inc.target, "title": inc.title,
             "report": inc.report, "status": inc.status, "findings": inc.findings, "actions": inc.actions}
        if inc.kind == "tbm_anomaly":
            d["symptoms"] = inc.symptoms
            d["cause_estimate"] = {k: round(v, 2) for k, v in
                                   tbm_posterior(inc.symptoms, inc.symptoms["est_cutter_wear"]).items()}
        if inc.where:
            d["where"] = inc.where
        if inc.status == "resolved":
            d["truth"] = inc.truth
            d["resolved_sol"] = inc.resolved_sol
        return d

    def observe(self) -> dict:
        """Everything a planner may see. No hidden truth."""
        p = self.p
        storm = None
        if self.storm:
            storm = {"since_sol": self.storm["start"], "sunlight_pct": round(100 * self.storm["factor"])}
        return {
            "sol": self.sol,
            "deadline_sol": p.deadline_sol,
            "sols_left": p.deadline_sol - self.sol,
            "goal": {"propellant_t": p.target_propellant_t, "pad_fill_m3": p.pad_fill_m3},
            "progress": {
                "propellant_t": round(self.propellant, 1),
                "pad_pct": round(100 * self.pad_done / p.pad_fill_m3, 1),
                "solar_kw": round(self.solar_kw),
                "plant": self.plant.status,
                "plant_build_pct": round(100 * min(1.0, self.plant.build_h / p.plant_build_crew_h), 1),
                "plant_throttle": self.plant.throttle,
            },
            "world": {"storm": storm, "earth_contact": self.comms_ok, "panel_dust_pct": round(100 * self.dust, 1),
                      "conjunction_sols": list(p.conjunction),
                      "resupply": {"landed": self.cargo_landed, "eta_sol": None if self.cargo_landed else self.cargo_arrival}},
            "power": self.last,
            "stocks": {"spares": dict(self.spares), "spoil_m3": round(self.spoil), "ore_at_portal_t": round(self.ore_portal),
                       "ore_at_plant_t": round(self.ore_plant), "crew_hours_owed": round(self.crew_debt, 1)},
            "route": {"name": self.route, "km": self.site.routes[self.route]["km"],
                      "rock_cover_pct": self.site.routes[self.route]["rock_cover_pct"],
                      "wheel_hits_per_round_trip": round(self.site.routes[self.route]["hits_per_trip"], 5),
                      "alternatives": [k for k in self.site.routes if k != self.route]},
            "machines": [
                {"id": t.id, "kind": "tbm", "tunnel": t.tunnel, "status": t.status, "down_reason": t.down_reason,
                 "paused": t.paused, "speed": t.speed, "position_m": round(t.position_m),
                 "access_m": t.access_m, "est_cutter_wear": round(t.est_wear, 2), "telemetry": t.telemetry,
                 "face": [round(x, 5) for x in self.site.tunnel_point(t.tunnel, t.position_m)],
                 "tile_odds": {k: self.site.tunnel_tile(t.tunnel, t.position_m)[k] for k in ("hard_per_km", "bearing_mult")},
                 "back_in_service_sol": t.repair_until if t.status == "repairing" else None}
                for t in self.tbms
            ] + [
                {"id": h.id, "kind": "hauler", "status": h.status, "down_reason": h.down_reason, "paused": h.paused,
                 "km": round(h.km), "back_in_service_sol": h.repair_until if h.status == "repairing" else None}
                for h in self.haulers
            ] + [{"id": "dozer", "kind": "dozer", "status": self.dozer.status, "down_reason": self.dozer.down_reason,
                  "paused": self.dozer.paused,
                  "back_in_service_sol": self.dozer.repair_until if self.dozer.status == "repairing" else None}],
            "open_incidents": [self.incident_view(i) for i in self.open_incidents],
            "outcome": self.outcome(),
        }

    # runs

    def advance(self, max_sols: int = 10_000) -> list[Incident]:
        """Step until an incident opens, the goal is met, or the deadline passes."""
        n = 0
        while not self.done and not self.open_incidents and n < max_sols:
            self.step()
            n += 1
        return self.open_incidents

    def clone(self) -> "Sim":
        c = copy.copy(self)
        c.p = self.p
        c.tbms = copy.deepcopy(self.tbms)
        c.haulers = copy.deepcopy(self.haulers)
        c.dozer = copy.copy(self.dozer)
        c.plant = copy.copy(self.plant)
        c.spares = dict(self.spares)
        c.storm = dict(self.storm) if self.storm else None
        c.incidents = copy.deepcopy(self.incidents)
        c.cargo_notice = dict(self.cargo_notice) if self.cargo_notice else None
        c.log = list(self.log)
        c.series = {k: list(v) for k, v in self.series.items()} if self.series is not None else None
        c.last = dict(self.last)
        return c
