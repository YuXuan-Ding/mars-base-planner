# Backend: algorithms and API

Design notes for the planner's backend (2026-10-17). The service lives in `backend/` (FastAPI, numpy, scipy).
**Built:** terrain layers, `/routes`, `/tunnels/plan`, `/sites/search` (least-flattening base layout), catalogs.
**Designed, not built yet:** spoil-limited flattening and the construction schedule.

```sh
python3 -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn app.main:app --app-dir backend --port 8001   # docs at /docs
backend/.venv/bin/python -m pytest -q backend/tests
```

## The core idea

The boring company's **tunnel boring machines (TBMs)** dig underground: to reach and extract minerals, and to
bore tunnels for transit or habitats. Everything they dig comes out as **spoil**, which is the dirt supply for
flattening pads and filling roads on the surface. So:

```
machines  ->  metres bored per sol  ->  ore (mining)  +  spoil m³  ->  how much ground we can flatten
                                      \->  tunnels as a third way to connect two places
```

## Terrain layers

The 200 m DEM is averaged into 2×2 blocks, giving a **~400 m working grid** (1007 × 1067 cells).
`MARS_GRID_FACTOR=1` uses the full 200 m grid but is slower.

| Layer | How | Status |
|---|---|---|
| Elevation | block mean of the USGS HRSC+MOLA 200 m DEM | real |
| Gradient, slope | central differences, metric spacing per row (cos lat) | real |
| Roughness | std of 200 m slope inside each cell | real |
| Crater rim distance | distance to the nearest Robbins crater rim | real (craters ≥ 1 km) |
| Rock abundance *k* | 0.04 background + crater ejecta + roughness term, clipped to 0.02–0.35 | **placeholder** |

Replace the rock layer with THEMIS/TES rock abundance (regional) or HiRISE rock counts (at sites). Nothing else
has to change.

**Resolution caveat.** At 400 m, the Jezero rim never gets steeper than about 7° along a move. Small scarps,
boulder fields and rim crests are smoothed away. Use the routes to compare options, and load 20 m CTX or
1 m HiRISE elevation for the chosen corridor before quoting earthwork.

## Equipment hazard requirements

Every vehicle carries its own limits. A route is **refused** wherever any one of them is exceeded:

| Limit | Meaning |
|---|---|
| `max_grade_deg` | steepest climb or descent along the direction of travel |
| `max_cross_slope_deg` | steepest sideways tilt (tip-over risk) |
| `max_rock_m` | largest rock the vehicle will drive over |
| `max_rock_cover_pct` | area is avoided if rocks bigger than `max_rock_m` cover more than this share of the ground (0 = avoid any measurable cover, below 0.1 %) |
| `min_crater_rim_clearance_m` | keep at least this far from crater rims |

**Fleet rule:** a mixed fleet travels together, so on each limit it is held to its strictest member. The
response says which vehicle set each limit (`binding_vehicle`). A request can override any limit fleet-wide with
`hazard_overrides`, for example "no rocks over 20 cm at all":
`{"max_rock_m": 0.2, "max_rock_cover_pct": 0}`.

**Rock size model.** Golombek & Rapp (1997) give the share of ground covered by rocks of diameter ≥ D, for a
surface with total rock abundance *k*:

```
F_k(D) = k · exp(−q(k) · D),   q(k) = 1.79 + 0.152 / k
```

For each cell we compute `F_k(max_rock_m)` and compare it with `max_rock_cover_pct`. A smaller rock limit, or a
lower cover tolerance, blocks more ground. With the default catalog on today's placeholder layer, the light
utility rover is blocked on about 4 % of the map, the crew rover on 14 %, the hauler on 21 % and the crane on 28 %.

## Routes (`POST /routes`)

**Graph.** 16-connected grid: king moves plus knight moves, so paths are not limited to 45° turns. Grade is
`|Δh| / length` along each move, not cell slope, so switchbacks emerge naturally. Cross slope is the terrain
gradient perpendicular to the move. A knight move also needs the two cells it passes between to be open.
Search runs in a window around the two endpoints (±40 km). If no accessible route is found there, it retries
on the whole map before reporting none. Endpoints on blocked cells snap to the nearest open cell, and the
snap distance is reported.

**Soft hazard cost.** Each cell gets a hazard score from 0 to 1: a weighted mean of slope ÷ grade limit,
rock cover ÷ rock limit, roughness ÷ p95, and crater-rim proximity (`exp(−d/2 km)`). The weights are adjustable
(`hazard_weights`). A move costs `length × (1 + avoidance × hazard)`. `avoidance` sets how far the route will
detour to stay on safer ground.

**Route kinds:**

1. **Accessible:** drivable today. Every move that breaks a fleet limit is removed, then Dijkstra.
   If the ends are disconnected, it returns `status: "none"` plus a **gap**: the closest approach between the area
   reachable from the start and the area reachable from the goal, the gap width, and which limits block it
   (`rocks`, `crater_rim`, `slope`, `cross_slope`).
2. **Shortest:** the cheapest road to build and use. Everything is allowed except cliffs (> 35°). Each move costs,
   in metres of driving: its length × (1 + 0.3 × hazard), plus `CLIMB_M / 2` = 4 m per metre of height change (a
   metre of climb costs about 8 m of level driving on loose ground, and a round trip climbs each height once),
   plus `earthwork_weight` × its grading (roadbed and bench). A move steeper than the fleet's grade is costed as
   the switchback road it would need: `|dh| / tan(max_grade) × 1.15` long, benched into the slope. So a high,
   steep mountain costs what crossing it really costs and the route goes around it when that is cheaper.
   `earthwork_weight` (m of road per m³) defaults to the grading time of 1 m³ divided by the driving time of 1 m
   over the horizon (`(1/grading_rate) / (trips × 2 / speed × 500 sols / hours per sol)`, about 0.25 m per m³ at one
   trip a sol), so building and driving are traded in the same sols. Along the chosen path the
   backend designs the **road line**: the lowest line that never sits below the ground and never exceeds the
   fleet's grade, so steep ground is built up with fill (ramps and embankments), pinned to the ground at both
   ends (cut only where that forces it). It is the upper envelope of grade cones from every point, two passes
   along the path. Earthwork:
   - fill (and cut): cross-section `w·d + d² / tan 35°` (road width plus side slopes at the angle of repose)
     at each point of depth `d`, integrated along the path; `design_profile` gives [km, ground, road]
   - roadbed under the whole road, not only the steep parts: `w × (0.3 m + tan(roughness) × w/4) × L`, graded and
     compacted across the full width (`roadbed_m3`; `road_area_m2` is the footprint `w × L`)
   - bench cut for any side slope (half cut, half fill): `w²/8 × tan cross × L` (`bench_m3`)
   - rock clearing on cells that fail the rock or rim limits

   It returns the earthwork m³, grading sols, rock-clearing km and sols, and a list of **work segments**
   (km range, m³, coordinates) for the map. The grading volume equals the fill needed, which TBM spoil can supply.

   **Steep stretches** (`steep`): every run where the road line leaves the ground by more than 1 m (runs within
   3 moves merge, plus one move each side for portals) is costed as a ramp or a road on the slope. Bores on steep
   stretches are off (`STEEP_BORES = False` in `app/routing.py`): they bored up a flank, drove across a flat top
   and bored down again, so boring is planned as a whole in the bore-through option instead, and each stretch's
   `bore` is `status: "none"`. With `STEEP_BORES` on, each stretch is costed three ways, and `recommended` picks the one with fewer
   equipment-sols (build + driving over the horizon, each metre climbed counted as 8 m). The route totals use the
   recommended picks. Neighbouring runs can also be crossed together by **one bore from the first run's foot to the
   last run's foot**, under the flatter ground between them (the two flanks of a mountain with a flat top, say)
   instead of a bore up one flank, a drive across the top and a bore down the other. The runs are grouped whichever
   way costs least overall (a small dynamic program over the runs; a group is bore-only and at most
   `max_tunnel_km` long).
   Once underground, a bore may stay there: each run's ends can grow outward over neighbouring moves that are
   still steeper than half the grade limit (`STAY_UNDER`), such as the drivable but sloping upper part of a
   mountain whose foot is the steep part. Grown ends are only ever bored. Three weights favour staying under:
   - **resurfacing penalty:** a bore that follows an earlier bore (the route came up and goes back down) costs
     `resurface_penalty_sols` extra (request field, default 100; 0 turns it off)
   - driving still-sloping ground right where a bore surfaces or dives counts 4× (`STAY_WEIGHT` = 3 extra)
   - every bore pays 15 sols of portal and TBM set-up (`setup_sols`, included in `bore_sols`)

   **Climbing or diving inside a bore** is weighted, up and down alike: a bore's cost counts each metre of height
   it travels (`vertical_m`: the rises and falls of its floor, plus any spiral or portal ramp between the floor's
   ends and the ground) as `tunnel_vertical_m` metres of tunnel (request field, default 20). So a level bore
   through a ridge is preferred, and a bore that only climbs a flank usually loses to a road. The same weight
   ranks the tunnel route: the first 12 buildable portal pairs, best first by score, are compared with
   `tunnel_weight × tunnel_vertical_m × vertical_m` added, and the cheapest is returned (`tunnel.vertical_m`).

   The search tries every option of every stretch (ramp, road, bore) with these weights included, so they can
   change the pick: two bores a few hundred metres apart become one, while bores kilometres apart stay separate.
   - **ramp:** the road line above, straight, with its fill and cut in that run plus its roadbed and bench
     (`max_fill_m` is the highest embankment).
   - **road on the slope:** long enough to gain the climb and drop at the fleet's grade
     (`(climb + drop) / tan(max_grade)`, +15 % for hairpins when it has to zigzag), cut as a bench into the
     hillside: `w² tan(side slope) / 8` per metre half cut / half fill, or `/ 2` (full cut) where the side slope
     is over 30° and fill won't hold.
   - **bore through:** a straight TBM bore between the stretch ends at the lower of the TBM's and the fleet's
     grade (vehicles drive it). Through a ridge it is the transit profile below. Up a scarp it climbs from the
     low portal while keeping its cover, then a spiral under the high portal gains the rest (`spiral_m`). It
     fails only if the ground between drops below the low portal.
3. **Least construction** (`minimal`): the route that needs the least building. Moves the fleet can already drive
   cost nothing to build; others cost their grading volume (switchback bench and roadbed, as for the shortest
   route) plus rock clearing as m³, and every metre of length costs 0.02 m³ (`MINIMAL_M3_PER_M`) so it doesn't
   detour without limit. Its road gets no roadbed or bench where the ground is already drivable, and its steep
   stretches are picked by build time alone. Same fields as `shortest`; `segments` are the stretches to build.
4. **Bore through** (`through`, `app/through.py`): one fixed route, the straight A–B line on the grid cells it
   crosses, and one decision, the **bore height** `height_m`. Every run of ground above it (300 m or longer) is a
   level TBM bore at that height, with portals where the ground crosses it; the rest is a built road on the line
   (`_build_road`, as `shortest`). Heights go from the higher of the two ends (`line.lowest_bore_m`) up to the
   highest ground (`line.highest_ground_m`); at or above that there is no bore. Scored in the shortest route's
   units: the shortest-route move costs along the road parts + each bore (boring and 15 set-up sols as metres of
   driving + its length + `tunnel_vertical_m` × the height its portal ramps make up) + `resurface_penalty_sols`
   for each bore after the first. A cliff on the road part (a move steeper than 35°) makes the height unbuildable
   (bore lower, below the cliff's top). `/routes` returns the rule's height: the lowest score over 80 heights, even when the shortest route
   scores better (`beats_shortest: false`). Fields: `height_m`, `bores` (km along the line, length, height,
   `bore_sols`, cover, profile), `legs` (the roads on the line), totals, `climb_m`/`drop_m`, `score_sols` and
   `shortest_score_sols`, `line` (km, ground, slope along the line). No bore may be longer than `max_tunnel_km`
   (default 40 km; about 100 sols with two machines).

   `POST /routes/through` takes the same body plus `decide`: `"ai"` (default) has Claude choose the height: it gets
   the line's elevation profile, the fleet, TBM and scoring, tries heights with an `evaluate_height` tool and
   submits one with its reasons (`decided_by`, `rationale`, `tries`). If Claude can't be reached, or submits a
   height that isn't buildable, the rule's height comes back with the reason. `"rule"` returns the rule's height;
   `height_m` costs a given height and `no_bore: true` the road over the top.

   The route recommendation (`recommended`) uses the same climb weights: build sols + driving over the horizon,
   each metre climbed or descended on the surface counted as 4 m and each metre up or down inside a bore as
   `tunnel_vertical_m` m.
5. **Tunnel:** drive, then portal, tunnel, portal, drive, with at most one straight TBM tunnel. Candidate
   portals sit on a ~2.5 km lattice of flat (≤ 5°) open cells reachable from each end. Each pair within
   `max_tunnel_km` is scored as `surface cost A→p + tunnel_weight × tunnel length + surface cost q→B`, and the
   best 120 pairs are checked for feasibility, best first, after a straight bore from A to B (tried first, so
   the tunnel runs end to end when it can; when it can't, `direct` says why, such as breaking out into a crater). The best buildable tunnel is returned even when driving
   scores better, flagged `beats_surface: false` with a `note` (it must still be shorter than the drivable
   route; otherwise `not_needed`). `beats_surface` is true when its score beats the drive and it cuts trip
   length by 10 % or more.

**Throughput.** `trips/sol = throughput ÷ payload of the largest hauling vehicle`; cranes and TBM carriers
don't count. More than 24 trips/sol gets a two-lane road, which doubles the width used for earthwork.
`haulers_needed = trips × round-trip hours ÷ operating hours per sol`, per route.

**Recommendation.** The option with the fewest equipment-sols: build time (grading + clearing + boring)
plus vehicle time driving the route over a 500-sol horizon.

## Tunnels (`POST /tunnels/plan`)

**Machine model** (`/catalog/machines`): diameter, advance in m/sol, duty factor, max grade, min rock cover,
swell factor.

```
bore rate   = advance × duty                      (4 m TBM: 333 m/sol × 0.6 ≈ 200 m/sol)
spoil       = π(D/2)² × length × swell            (≈ 3,500 m³ of loose spoil per machine-sol)
```

A boring machine is a third as fast as a construction unit, which grades 600 m of road a sol on flat ground
(`construct/model.py`). The compact 2 m TBM bores 585 m/sol × 0.65 ≈ 380 m/sol.

```
ore         = π(D/2)² × drift length × ore fraction × rock density
```

**Profile.** The tunnel is straight in plan. Its floor is the deepest line the grade limit allows from each
portal, so cover is as large as possible:
`z(s) = max(h_a − g·s, h_b − g·(L − s), lowest surface − cover)`. If that line would come out above the surface
anywhere, for example under a valley, the result is `status: "breakout"` with the distance. Shallow runs
touching a portal are reported as `portal_cut_m` (open approach cuts). Shallow stretches anywhere else are
reported as `shallow_m`.

**Purposes:**

- **mining:** portal on the flattest ground near `from`; decline toward a point `depth_m` under `to`, with a
  spiral `decline_m` if the straight run is too short to get that deep; then `drift_length_m` of branch tunnels
  through the ore. One machine drives the access tunnel; all of them work the drifts.
- **habitat:** same access, then a bored chamber of `chamber_length_m`.
- **transit:** portals at both ends, boring from both faces when there are two or more machines.

**Assumptions to replace with data:** there is no subsurface data, so ore depth, ore fraction, rock density
and hardness (and with it the advance rate) are request inputs with stated defaults. Every response lists its
assumptions.

## Base site search (`POST /sites/search`, built)

Send an area (`area`: two corners, or `polygon`: any outline, where only pads that fit wholly inside count) and the facilities the base needs; get back where they fit with the least ground to
flatten. Facility presets are at `/catalog/facilities`, each with the sources behind its size (`basis`):

| Facility | Pad | Max tilt | Separation | Basis |
|---|---|---|---|---|
| Starship landing pad | 100 × 100 m | 2° | 1 km from everything | prepared pads ≤ 100 m per Starship payload (NTRS 20220000334); ships ~1 km apart, 1 km blast radius (Heldmann 2022; Metzger 2021) |
| Habitat area | 60 × 60 m | 3° | | assumed; crews first live in landed Starships (Heldmann 2022) |
| Propellant plant | 100 × 100 m | 3° | | assumed; Sabatier + electrolysis, 1,200 t of CH4/LOX per refill (Heldmann 2022) |
| Solar farm, 1 MW | 200 × 200 m | 8° | | ~40,000 m² per MW averaged over the sol (after Zubrin); a two-ship plant needs ~2.5 MW |
| Water ice extraction | 50 × 50 m | 10° | | assumed Rodwell site; ~600 t of water per refill (Heldmann 2022) |
| Cargo and storage yard | 50 × 50 m | 5° | | assumed |

**Algorithm.** On the native 200 m grid:

1. **Cost map per facility.** For a pad centred on every cell, fit a plane with summed-area tables of `h, xh, yh,
   h²` (constant time per position, window at least 3 × 3 cells). Earthwork (cut + fill, balanced) =
   `area × 0.8 σ` (bumpiness left after the plane, scaled down for pads smaller than the window) +
   `max(0, |grad| − tan(max tilt)) × side³ / 4` (levelling the tilt beyond the allowance).
2. **Greedy layout per candidate centre** (up to 600 centres on a grid over the area): largest pad first, cheapest
   free spot within `max_spread_km`, squares never overlap (30 m gap), landing pads keep 1 km from everything
   (their search radius grows by that separation). Ties go to flatter ground, then closer to the centre.
3. **Rank** centres by total earthwork; return the best layout and alternatives at least the spread apart.

At 200 m, bumps smaller than a few hundred metres are invisible, so flat crater floor reads as near-zero
earthwork: use it to rank sites, then load CTX or HiRISE elevation at the winner. `"trace": true` adds the cost
maps (thinned to 90 × 90, log-scaled), every centre in evaluation order with its score and rank, and the layout
steps, which the web map replays.

`GET /terrain/patch?lon=&lat=&half_km=` returns the native 200 m elevation around a point (base64 int16), which the
map loads when zoomed in.

## Base construction with spoil (designed, not built yet)

**Flattening with a spoil supply.** For each pad, fit the best plane whose tilt stays within the facility's
`max_slope`. With no outside dirt, the least-squares plane balances cut and fill. With spoil available (S m³),
raise the pad height z until `fill(z) − cut(z) = S`, using binary search because fill rises and cut falls
with z. More spoil means less surface digging. Add haul distance from the portal to each pad as a cost.

**Fast site screening.** Summed-area tables of `h, h², xh, yh, x², xy, y²` give the plane fit and residual spread for any window in
constant time, so a whole-map "cost to flatten here" layer takes one pass per footprint size. Exact volumes are
computed only for the top ~50 candidates.

**Several facilities.** One cost map per facility type, then greedy placement with restarts under constraints:
landing pad ≥ ~1 km from the habitat (blast ejecta), everything within R km, solar on sunlit ground.

**Scale check.** A 400 × 400 m pad needing 0.5 m of fill on average takes 80,000 m³, about 950 machine-sols of
spoil. The dirt supply will be the binding limit, so prefer naturally flat sites and use each facility's
slope tolerance.

## Planned endpoints

```
POST /sites/search     machines, facilities [{type, area_m2, max_slope}], spoil source, constraints
                       -> ranked candidates, per-pad layout, fill from spoil, haul km, sols
POST /sites/evaluate   same for a clicked site
POST /plan/schedule    machines + jobs (tunnels, pads, routes) -> timeline and a per-sol spoil ledger
                       (produced, used, stockpiled). This is where the agent reasons across the plan.
```

## API reference (built)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/terrain/meta` | bounds, resolutions, which layers are real |
| GET | `/terrain/sample?lon=&lat=` | layer values at a point |
| GET | `/catalog/machines` | TBMs |
| GET | `/catalog/equipment` | surface vehicles with their hazard limits |
| POST | `/routes` | accessible / shortest / tunnel routes for a fleet |
| POST | `/tunnels/plan` | mining, habitat or transit tunnel |
| GET | `/catalog/facilities` | base facility presets with sources |
| POST | `/sites/search` | least-flattening base layout in an area |
| GET | `/terrain/patch?lon=&lat=&half_km=` | native 200 m elevation around a point |

Example request:

```json
POST /routes
{
  "from": [77.45, 18.45], "to": [78.2, 18.4],
  "equipment": ["hauler", "crane"],
  "throughput_t_per_sol": 200,
  "hazard_overrides": {"max_rock_m": 0.2, "max_rock_cover_pct": 1},
  "modes": ["accessible", "shortest", "tunnel"],
  "tbm": {"model": "tbm-4m", "count": 2}
}
```

The response has `fleet` (effective limits, binding vehicle, road width, lanes, trips), one block per mode, and
`recommended`. Paths are GeoJSON LineStrings in `[lon, lat]`. Each path also carries a `profile` of
`[km, elevation]` pairs and a `hazard` summary: max and mean score, km above 0.5, max rock cover, and closest
crater rim.

`"trace": true` adds a `trace` block for visualizing the calculation: the search window, a sample of cells
the fleet limits close (with the reason: 1 rocks, 2 crater rim, 3 slope), up to 8,000 explored cells in the order
Dijkstra settled them (`[lon, lat, cost ÷ final cost]`; sorting the distance field replays the search exactly),
counts for each stage, and for tunnels the candidate portals and every portal pair checked with its result.
The web map uses it to replay the calculation step by step.

Custom equipment can be sent inline instead of a catalog id:
`{"id": "my-rover", "name": "...", "width_m": 3, "payload_t": 1, "speed_kmh": 5, "limits": {...}}`.

## Next

1. Spoil-limited flattening in `/sites/search`, then `/plan/schedule` with the spoil ledger.
2. Wire the frontend: route drawing, gap and work-segment highlights, tunnel profile chart.
3. Real rock abundance (THEMIS/TES), then CTX elevation in chosen corridors for grading volumes.
4. Tunnels with a bend (two segments) and multi-tunnel routes; habitat chambers off the mining decline.
