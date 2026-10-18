# Mars Base Planner

A hackathon project for choosing where to build a Mars base and how to connect it. It uses real terrain from the Jezero crater region (about 405 × 403 km, 15.1–21.9° N, 73.4–80.6° E) to find flat, low-hazard sites, estimate grading work, and plan rover routes.

Planning notes, region choice and the data inventory are in [docs/brainstorm.md](docs/brainstorm.md).

## What's here

```
web/index.html        Interactive prototype (WebGL particle map, layers, bases, overview stack)
web/terrain.js        Packed terrain grid + craters + traverse + backdrop, built by scripts/build_terrain.py
backend/              FastAPI service: hazard-aware routes, TBM tunnels (see docs/backend.md)
scripts/fetch_dem.py  Crops the USGS HRSC+MOLA 200 m global DEM to our box via HTTP range requests
scripts/fetch_context.py  Coarse (~7 km) elevation for a 5x wider backdrop around the box, same DEM
scripts/build_terrain.py  Computes slope/roughness and packs the web data
data/                 Elevation crops (int16 meters) + metadata; raw crater and traverse downloads
docs/                 Brainstorm and planning notes
```

## Run it

```sh
cd web && python3 -m http.server 8000
# open http://localhost:8000
```

Backend API (routes and tunnels):

```sh
python3 -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn app.main:app --app-dir backend --port 8001   # API docs at http://localhost:8001/docs
(cd backend && .venv/bin/python -m pytest -q tests)
```

## Rebuild the data

Both scripts use only the Python standard library.

```sh
python3 scripts/fetch_dem.py      # ~9 MB download, writes data/jezero_dem_200m.*
python3 scripts/fetch_context.py  # ~6 MB download, writes data/context_dem.* (optional backdrop)
python3 scripts/build_terrain.py  # writes web/terrain.js
```

## Planner inputs

In **Overview**, each dataset sheet has a Keep / Drop switch. Dropped sheets sink to the bottom of the stack, turn grey, and leave the planner. Back on the map, everything except routes is computed from the kept layers only: the hazard score is reweighted over the kept inputs (slope and roughness, crater rims, soft ground), and base cards list any numbers they can't compute without a dropped layer. The choice is saved in the browser.

The backend and agent can read the same state from the page:

```js
PLANNER.config()   // { region, inputs, dropped, hazard: [{input, term, weight}], bases: [{..., stats}] }
addEventListener('planner:change', e => e.detail)  // fires when inputs or bases change
```

## The three steps: plan, route, build

The Bases panel walks through three steps strictly in order: **1 Plan** (choose sites; the bases), **2 Route**
(between two bases) and **3 Build** (construct the chosen route). Route unlocks once there is a base; Build unlocks
only when a route has been chosen. Going back a step closes what the later steps had open, since a change to the plan
can invalidate the routes.

A base is a group of facilities planned together: everything one site search places becomes one base (Alpha base,
Bravo base, …) whose facilities are linked by default by internal roads (the shortest set of straight links joining
them, dashed on the map). The base card lists its facilities and the internal road length; expand it for each
facility's numbers.

## Routes between bases

In step 2, choose **From** and **To** (two bases, or a base and a point picked on the map), the fleet, haul rate and
number of boring machines, and **Find routes**. Routes run from the facility nearest the middle of each base. That
sends the two ends and a fleet to the planner backend
(`POST /routes`, default `http://localhost:8001`, override with `?api=http://host:port`). Routes always weigh
elevation, slope and the full hazard score (slope, roughness, crater rims, rocks), whatever layers are shown or
kept in Overview. The calculation panel then replays what the backend did, step by step,
with a timeline you can scrub: search window, ground the fleet can't cross, the Dijkstra search spreading from the
start, the drivable route or the gap that blocks it, the shortest route with its earthwork stretches, tunnel portals
and the pairs tried, and the recommendation with an elevation or tunnel profile.

The shortest route is the cheapest road to build and drive over 500 sols: climbing costs (each metre of climb counts as
about 8 m of level driving) and ground steeper than the fleet can drive is costed as the switchback road it would
need, so it goes around a high, steep mountain when that is cheaper than crossing it.

The tunnel option is shown whenever a buildable tunnel exists, marked "marginal" when driving is better. The
shortest route is a built road as wide as the widest vehicle plus 1 m: its earthwork always includes a 0.3 m roadbed graded
under its whole length and a bench cut wherever the ground tilts sideways. It also designs its road line at the fleet's grade limit and reports the fill needed to build ramps and
embankments where the ground is steeper (the elevation chart shows ground and road line). Every such stretch gets a
choice under **Steep stretches**: **fill a ramp** (straight, on an embankment) or a switchback **road on the slope** (cut
into the hillside). Boring is planned only in the **Bore through** option below, by Claude or the rule, so the
shortest route never bores up a flank, drives across a flat top and bores down again. (Steep-stretch bores are
still in the code behind `STEEP_BORES` in `backend/app/routing.py`, off by default; with it on, a stretch can also be
bored, as follows.) Two steep flanks with flatter ground between them (a mountain with a flat top) can share one straight bore
from foot to foot instead of a bore up, a drive across the top and a bore down. A bore also tends to stay underground: it can
carry on through ground that is still sloping (over half the fleet's grade limit) instead of surfacing as soon as the
ground is drivable, each bore pays 15 sols of portal and machine set-up, and coming up out of one bore to go down into another costs a
further 100 sols (`resurface_penalty_sols` on `POST /routes`), so close bores join into one. Going up or down inside a tunnel
is weighted too, both directions the same (each metre of height counts as 20 m of tunnel, `tunnel_vertical_m`), so
level bores are preferred and a bore that only climbs a slope usually loses to a road.
Bores are drawn on the map at the same ×20 vertical exaggeration as the terrain. Tunnels try a straight bore from end to end first; the tunnel chart shows the whole trip. Each shows length, earthwork or boring sols, and spoil; the backend marks the cheaper one, and
switching updates the route totals and the map (orange zigzag for a road, dashed bore underground).

Two more options sit beside them:

- **Least construction** builds only where the fleet can't already drive (steep, rocky or rim ground) and drives
  the rest as it is, accepting a longer route; construction grades only those stretches.
- **Bore through** keeps one fixed route, the straight A–B line, and decides only the **bore height**: wherever the
  ground along the line rises above it, the road goes into a level bore at that height, and the rest is a built road
  on the line. A lower height means less climbing and descending and a longer bore; a height above the highest
  ground means no bore. By default the backend picks the height with the lowest score. **Ask Claude to plan the
  bore** sends Claude the elevation along the line; Claude tries heights (each costed by the backend) and picks
  one, or no bore, and the panel shows its reasoning and every height it tried. The route never changes, only how
  much of it is underground. A boring machine is a third as fast as a construction unit (the main TBM bores 200 m
  a sol), so a level bore under a high mountain usually beats the road around it.

## Planning a base

**Plan a base** (step 1 in the Bases panel): set how many of each facility the base needs (presets from the backend, sized from
SpaceX Starship mission studies; see `docs/backend.md`), change the ground each one needs, then **Mark search area**
and draw around the ground the base could use: drag freehand (a lasso closes itself), or click corner by corner and
click the first point or press Enter (Backspace undoes a corner). Only pads that fit wholly inside the outline count. **Find the best site** asks the backend for the layout that needs the least
flattening. The calculation panel replays it: a cost map per facility (green = little to flatten), every candidate
base centre scored, the greedy layout at the winner, then the best site and alternatives highlighted on the map.
**Add these facilities to the plan** turns the layout into one base. **Clear all** in the top bar (click twice to confirm)
removes every base, the search outline, the picked route end and any route or site result.

## Building it: the construction simulator

Each route option in the route panel has a **Choose** button, which takes you to step 3. There, set the number of
robotic units and who decides incidents, and **Start construction**. It runs the construction simulator
(`backend/construct`, see [docs/construction.md](docs/construction.md)) on exactly that line plus the internal roads
of the bases at both ends: the roads are split into 400 m tiles, and steep, rocky tiles take longer and cause more
incidents. Incidents are decided by **Claude**, **Me** (Claude can still be asked per incident) or a **fixed
rule**. The Construction panel plays the job back on the map (grey tiles to do, orange being worked, green done,
units as labelled dots) with progress, spend against budget, sol against deadline and each unit's status. The speed
slider goes from 0 (stopped) to 200 sols per second.

Bores on a tunnel or bore-through route are simulated with the road. The boring machines (the **Boring machines**
count from step 2) set up at the portals for 15 sols, then bore tile by tile toward each other, at a third of a
construction unit's speed (200 m a sol each). They are the blue squares (T1, T2) moving along the dashed bore line. A bore
splits the road: the crews start on both sides, in proportion to the work there, and none crosses until the bore breaks
through. Dust storms don't stop boring; TBMs have their own incident, a cutterhead fault. Work the simulator doesn't
model takes a fixed time after the road: a pad at each base end (10 sols). The report lists it as fixed and gives the
total.

At each incident (breakdown, boulder, slope failure, stuck unit, dust storm) playback stops on an incident card with
the options and their estimated delay and cost. In Claude mode the page asks Claude (`claude-opus-5-5`) and shows its
choice and reasoning; in Me mode you pick, or press **Let Claude decide**. Every decision goes in the log with who made
it. When the job finishes, the report compares plan, target and actual time and cost for the road.

**Ask Claude** in the top bar opens a chat about the project (planner, routes, simulators). While a job is open,
questions include its state unless you untick "about the construction job". It needs `ANTHROPIC_API_KEY` in
`backend/.env`.

Zoom in and the terrain turns solid gradually, from the middle of the view outward: the lit surface fades in over the particles near where you are looking and dissolves back into particles farther out, and the solid area grows as you zoom (it starts around the whole-region view and fills the screen by about 140 km across). The surface has contour lines (20–100 m apart depending on zoom) and is tinted lighter on high ground within the view. Closer in, an ~80 km patch of the backend's full 200 m elevation takes over around the view without a visible edge: its rim sits on the coarse 0.84 km surface and the 200 m relief ramps in over the outer 14 km, and when the patch loads or moves with the view, its relief grows out of the coarse surface over about a second. **Solid ground: on/off** in the top bar turns the solid surface off (and back on), leaving the terrain as particles at every zoom; the choice is remembered in the browser. **NASA fonts: on/off** switches titles to a wide, worm-logo-style face (Michroma) and text to Helvetica, as in NASA's 1975 Graphics Standards Manual; map labels follow, and the choice is remembered too. **Sound: on/off** plays an ambient soundtrack made in the browser with Web Audio (no audio files): thin Martian wind, a low drone and a faint chime now and then. It starts off, since browsers only allow sound after a click; if it was on last time, the first click on the page brings it back. It pauses while the tab is hidden.

## Camera

Drag to orbit (turn and tilt). Right-drag, Shift-drag or two fingers pan. WASD or the arrow keys move, Q/E turn, R/F tilt, +/- zoom. Scroll zooms toward the point under the cursor. Up close the camera will not dip into the hills.

The map stays flat and goes on in every direction: past the planning box is the ~7 km backdrop (about 2,000 km across), and past that a coarser ~30 km field from MOLA's 1/4° global elevation (`scripts/fetch_globe.py` writes `web/mars_globe.bin`) that runs for thousands of km. Zoom out stops a little past the whole planning region, and the camera can't tilt up past 32° below level, so there is never a horizon. The farthest ground fades into a dark haze instead. The planning particles are drawn at 1.6 per cell of flat area (scaled up on slopes); up close the solid surface carries the detail.

Use **Center view on** in the Bases panel, click a base card, or click a base's marker to lock the orbit onto that base. **Reset view** goes back to the whole region.

The raw crater and traverse files in `data/raw/` came from:

- `https://astrogeology.usgs.gov/pygeoapi/collections/mars/robbinsv1/items?f=json&bbox=73.4,15.1,80.6,21.9&limit=20000`
- `https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_traverse.json` (and `M20_waypoints.json`)

## Data status

Real: elevation (USGS HRSC + MOLA blended, 200 m), slope and roughness derived from it, Robbins craters, the Perseverance traverse.

Placeholders for now: rock abundance (modeled from crater ejecta and roughness), thermal inertia, minerals, geologic units, and the CTX/HiRISE coverage outlines.
