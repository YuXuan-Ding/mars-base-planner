# Mars Base Planner

A hackathon project for choosing where to build a Mars base and how to connect it. It uses real terrain from the Jezero crater region (about 405 × 403 km, 15.1–21.9° N, 73.4–80.6° E) to find flat, low-hazard sites, estimate grading work, and plan rover routes.

Planning notes, region choice and the data inventory are in [docs/brainstorm.md](docs/brainstorm.md).

## What's here

```
web/index.html        Interactive prototype (WebGL particle map, layers, bases, overview stack)
web/terrain.js        Packed terrain grid + craters + traverse, built by scripts/build_terrain.py
scripts/fetch_dem.py  Crops the USGS HRSC+MOLA 200 m global DEM to our box via HTTP range requests
scripts/build_terrain.py  Computes slope/roughness and packs the web data
data/                 Elevation crop (int16 meters) + metadata; raw crater and traverse downloads
docs/                 Brainstorm and planning notes
```

## Run it

```sh
cd web && python3 -m http.server 8000
# open http://localhost:8000
```

## Rebuild the data

Both scripts use only the Python standard library.

```sh
python3 scripts/fetch_dem.py      # ~9 MB download, writes data/jezero_dem_200m.*
python3 scripts/build_terrain.py  # writes web/terrain.js
```

The raw crater and traverse files in `data/raw/` came from:

- `https://astrogeology.usgs.gov/pygeoapi/collections/mars/robbinsv1/items?f=json&bbox=73.4,15.1,80.6,21.9&limit=20000`
- `https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_traverse.json` (and `M20_waypoints.json`)

## Data status

Real: elevation (USGS HRSC + MOLA blended, 200 m), slope and roughness derived from it, Robbins craters, the Perseverance traverse.

Placeholders for now: thermal inertia, minerals, geologic units, and the CTX/HiRISE coverage outlines.
