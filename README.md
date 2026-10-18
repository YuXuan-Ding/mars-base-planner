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
backend/.venv/bin/python -m pytest -q backend/tests
```

## Rebuild the data

Both scripts use only the Python standard library.

```sh
python3 scripts/fetch_dem.py      # ~9 MB download, writes data/jezero_dem_200m.*
python3 scripts/fetch_context.py  # ~6 MB download, writes data/context_dem.* (optional backdrop)
python3 scripts/build_terrain.py  # writes web/terrain.js
```

## Planner inputs

In **Overview**, each dataset sheet has a Keep / Drop switch. Dropped sheets sink to the bottom of the stack, turn grey, and leave the planner. Back on the map, everything is computed from the kept layers only: the hazard score is reweighted over the kept inputs (slope and roughness, crater rims, soft ground), and base cards list any numbers they can't compute without a dropped layer. The choice is saved in the browser.

The backend and agent can read the same state from the page:

```js
PLANNER.config()   // { region, inputs, dropped, hazard: [{input, term, weight}], bases: [{..., stats}] }
addEventListener('planner:change', e => e.detail)  // fires when inputs or bases change
```

## Routes between sites

In the Bases panel, **Connect two sites** and **Plan a base** are dropdowns: closed, each shows a one-line summary
(the two ends, or the facility count and search area), and the page remembers which ones you left open.

**Connect two sites** sends two bases (or a picked map point) and a fleet to the planner backend
(`POST /routes`, default `http://localhost:8001`, override with `?api=http://host:port`). Kept/dropped layers in
Overview set the backend's hazard weights. The calculation panel then replays what the backend did, step by step,
with a timeline you can scrub: search window, ground the fleet can't cross, the Dijkstra search spreading from the
start, the drivable route or the gap that blocks it, the shortest route with its earthwork stretches, tunnel portals
and the pairs tried, and the recommendation with an elevation or tunnel profile.

The tunnel option is shown whenever a buildable tunnel exists, marked "marginal" when driving is better. The
shortest route is a built road as wide as the widest vehicle plus 1 m: its earthwork always includes a 0.3 m roadbed graded
under its whole length and a bench cut wherever the ground tilts sideways. It also designs its road line at the fleet's grade limit and reports the fill needed to build ramps and
embankments where the ground is steeper (the elevation chart shows ground and road line). Every such stretch gets a
choice under **Steep stretches**: **fill a ramp** (straight, on an embankment), a switchback **road on the slope** (cut
into the hillside), or a TBM **bore through** (straight through a ridge, spiralling up inside the hill on a one-way
climb). Tunnels try a straight bore from end to end first; the tunnel chart shows the whole trip. Each shows length, earthwork or boring sols, and spoil; the backend marks the cheaper one, and
switching updates the route totals and the map (orange zigzag for a road, dashed bore underground).

## Planning a base

**Plan a base** in the Bases panel: set how many of each facility the base needs (presets from the backend, sized from
SpaceX Starship mission studies; see `docs/backend.md`), change the ground each one needs, then **Mark search area**
and draw around the ground the base could use: drag freehand (a lasso closes itself), or click corner by corner and
click the first point or press Enter (Backspace undoes a corner). Only pads that fit wholly inside the outline count. **Find the best site** asks the backend for the layout that needs the least
flattening. The calculation panel replays it: a cost map per facility (green = little to flatten), every candidate
base centre scored, the greedy layout at the winner, then the best site and alternatives highlighted on the map.
**Add these facilities to the plan** turns the layout into bases. **Clear all** in the top bar (click twice to confirm)
removes every base, the search outline, the picked route end and any route or site result.

Zoom in and the terrain turns solid gradually, from the middle of the view outward: the lit surface fades in over the particles near where you are looking and dissolves back into particles farther out, and the solid area grows as you zoom (it starts around the whole-region view and fills the screen by about 140 km across). The surface has contour lines (20–100 m apart depending on zoom) and is tinted lighter on high ground within the view. Closer in, an ~80 km patch of the backend's full 200 m elevation takes over around the view without a visible edge: its rim sits on the coarse 0.84 km surface and the 200 m relief ramps in over the outer 14 km, and when the patch loads or moves with the view, its relief grows out of the coarse surface over about a second.

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
