# Brainstorm and planning notes

Working notes from the hackathon kickoff (2026-10-17). This records what we looked into, what we decided, and why.

## Concept

A Mars geographic planning tool that helps users decide where to build bases and how to connect them, using real Mars terrain data to:

- find flat, low-hazard areas for a base
- estimate how much terrain needs grading (cut and fill)
- compare candidate sites by their geography
- compute slope- and hazard-aware routes between bases and resources
- show how route feasibility changes with rover limits (max slope, range)

The demo should spend most of its time on the **business logic and agentic AI**, not on processing the whole planet. So we work on one mid-size region.

## How realistic is simulating slope and hazard?

**Slope** is straightforward and grounded in real data: take the gradient of an elevation model (Horn's method, or `gdaldem slope`).

- **Resolution changes the answer.** At 463 m/px a 30° crater wall can show up as 5°. Use coarse data for the regional view and fine data for site work.
- Pixel spacing must be in meters, not degrees. In a lat/lon grid, the east-west spacing shrinks with cos(latitude).

**Hazard** is a composite score we design ourselves from real layers:

| Input | Source | Why it matters |
|---|---|---|
| Slope | Elevation model | Rover and construction limits |
| Roughness | Local spread of slope or elevation | Catches broken ground that average slope misses |
| Crater rims | Robbins crater database | Steep and rocky edges |
| Soft ground | THEMIS thermal inertia (low = dust or sand) | Sinkage risk |
| Rocks | TES rock abundance, HiRISE rock counts | Boulders |

We normalize each input to 0–1 and take a weighted sum. The weights could be exposed as sliders. We present the result as a planning heuristic, not a physics simulation.

**How it feeds the other features:**

- **Site finder:** slide a facility-sized window over the map and keep spots with low max slope and low roughness.
- **Cut and fill:** fit a level pad to the footprint. The volume moved is the sum of |elevation − pad height| × pixel area.
- **Routes:** build a cost grid (distance × hazard, impassable above the max slope) and run A* or Dijkstra (`skimage.graph.route_through_array`).
- **Rover limit slider:** re-run with a different max slope and show the route changing or becoming impossible. Curiosity is rated to about 30° of tilt; operations stay nearer 15–20°.

## ML / computer vision options for hazard detection

| Dataset | What it is | Use for us |
|---|---|---|
| DoMars16k | ~16k labeled CTX tiles, 15 landform classes | Sliding-window terrain-type layer |
| HiRISE landmark dataset (JPL, Zenodo) | ~10k HiRISE crops, ~8 classes | Same idea, higher resolution |
| Crater detection sets (Robbins + CTX/THEMIS; YOLO sets on Roboflow/Kaggle) | Crater boxes | Finds craters under 1 km that the database misses |
| Segment Anything (SAM) | Zero-shot segmentation | Boulders and rims at one HiRISE site |
| AI4Mars (JPL) | ~35k rover navcam images labeled soil, bedrock, sand, big rock | Rover-eye view demo. It doesn't map onto orbital images. |

JPL's SPOC model classified HiRISE terrain for traversability on the Mars 2020 rover mission, which supports this approach. Its weights aren't public.

Plan if time allows: a YOLO crater detector on CTX, plus a DoMars16k classifier. Both outputs feed the hazard score next to slope and roughness. The story for the demo: elevation tells us how steep a place is, ML on images tells us what it is.

## Choosing a region

The guiding principle: **pick an area where the decisions are hard.** If everything is flat, every answer is obvious. If everything is rough, there is no answer.

Criteria, in priority order:

1. Ready elevation data (CTX ~20 m, ideally HiRISE ~1 m at base sites).
2. A mix of terrain inside the box: plains for bases, plus rims, scarps, mesas and channels as obstacles.
3. Several candidate sites with different trade-offs.
4. Destinations to route to: water ice, minerals, natural shelter.
5. Varied grading difficulty between sites.
6. Size: we settled on 200–500 km across.
7. A name judges recognize.

### Environmental factors (dust, pressure, radiation)

Over a few hundred km, most of these are nearly uniform. They matter for **choosing the region and for timing**, not for picking between sites inside it.

| Factor | Varies inside the region? | Use |
|---|---|---|
| Dust storms | No, regional | Seasonal planning |
| Air pressure | Somewhat, via elevation (about 17% more per 2 km lower) | Landing ease, shielding |
| Radiation | Somewhat (elevation, cliffs blocking sky) | Shelter siting |
| Sunlight / solar | Yes (shadows from walls and slopes) | Solar farm placement |
| Dust devils, local wind | Yes | Panels and equipment |
| Dust cover / loose soil | Yes | Already in the hazard layer |

The link to business logic: dust lowers solar power, which cuts machine hours and lengthens the grading timeline. Example agent reasoning: "Start grading the landing pad before dust season."

Sources: Mars Climate Database (modeled), Perseverance MEDA and Curiosity REMS weather stations, Curiosity RAD (the only surface radiation measurement, ~0.67 mSv/day at Gale), NASA SWIM ice maps.

### Candidates at 200–500 km

| Region | Strengths | Gaps |
|---|---|---|
| **A: Jezero / Nili Fossae / NE Syrtis** (~350–400 km) | Continuous HRSC 50 m elevation, CTX 20 m, HiRISE 1 m mosaics; Perseverance MEDA weather; varied terrain; rich mineral targets; real rover path to compare against | No radiation measurements, no shallow ice |
| B: Gale Crater to InSight (~550 × 250 km) | Two weather stations, RAD radiation, InSight seismic data | Fine elevation only in patches; slightly over 500 km |

**Decision: Region A.** Our core features depend on terrain detail. For the environment, modeled values plus one weather station are enough. For radiation, we scale Curiosity's measured dose by elevation and say so openly.

At this size we work at two levels: a regional grid (~150–200 m) for the map, routes and screening, and detailed CTX/HiRISE data loaded only around sites the agent picks.

## Data inventory for Region A

| Group | Dataset | Resolution | Status in the prototype |
|---|---|---|---|
| Terrain | USGS HRSC + MOLA blended DEM v2 | 200 m | **Loaded** (cropped by HTTP range requests) |
| Terrain | HRSC MC-13 regional DTM tiles | 50 m | Not yet |
| Terrain | USGS CTX DEM mosaic over Jezero | 20 m | Not yet |
| Terrain | USGS Mars 2020 TRN HiRISE DTM + ortho mosaics; MSR TRN mosaics (2025) | ~1 m | Not yet |
| Hazard | Robbins crater database (USGS pygeoapi) | ≥ 1 km | **Loaded** (490 craters) |
| Hazard | THEMIS thermal inertia | ~100 m | Placeholder |
| Hazard | TES dust cover and rock abundance | coarse | Not yet |
| Geology | USGS SIM 3464, Jezero and Nili Planum geologic map | 60 × 60 km | Placeholder |
| Resources | CRISM mineral maps; Mars 2020 ellipse mineral map (Zenodo) | ~18 m, patches | Placeholder |
| Environment | Perseverance MEDA (PDS); Mars Climate Database | n/a | Not yet (pressure estimated from elevation) |
| Ground truth | Perseverance traverse and waypoints GeoJSON (NASA MMGIS) | path | **Loaded** (through sol 2009) |

The traverse URLs come from a community post and are unofficial:
`https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_traverse.json` and `.../M20_waypoints.json`.

## Grading time model

- **Volume:** cut-and-fill volume from elevation over each facility footprint (size and slope tolerance come from user input).
- **Machine profile:** rate (m³/hour or per sol), power draw, soil-type multiplier. Start from Earth equipment specs scaled for 0.38 g, with NASA's RASSOR excavator as a reference point.
- **Time:** volume ÷ rate, adjusted for working hours (dust, power), then sum and schedule across facilities. This is the kind of multi-step plan the agent can show off.
- **Open question:** "boring machine" means tunneling, not flattening. If underground habitats are wanted, they need a separate estimate.

In the prototype, cut ≈ L³·tan(slope)/8 + L²·roughness term, and the grading rate is a user input (default 50 m³/sol, assumed).

## Interface design

- **Default view:** a calm, near-black WebGL particle map of the region's elevation (~230k points on a 0.84 km grid, ×20 vertical exaggeration). Particles rise on load and the view drifts slowly when idle.
- **Layers panel:** every dataset, with toggle, legend, opacity, and its source and resolution. Each is marked real or placeholder.
- **Bases panel:** facility type, footprint, max slope and grading rate, then click the map to place. Each card shows elevation, slope against tolerance, hazard, cut and fill, grading time in sols, pressure, and nearest mineral.
- **Overview mode:** an exploded stack with one floating sheet per dataset over the same box (elevation → slope → thermal inertia → hazard → minerals → geology → craters → coverage → your plan). Dashed guides run through Jezero and each base.

## Next steps

1. Slope-limited routing between bases, with a rover max-slope slider and a range limit.
2. Replace the placeholder layers with real THEMIS, CRISM and USGS geologic data.
3. Load CTX and HiRISE detail around chosen sites for real cut-and-fill.
4. Site-finder search, plus the agent layer that compares sites and schedules construction.
5. Optional ML: crater detector and terrain classifier.

## Sources

- [HRSC HMC-30 regional DTMs](https://hrscteam.dlr.de/public/data/regionaldtms.php)
- [USGS Mars 2020 TRN HiRISE DTM mosaic](https://www.usgs.gov/data/mars-2020-terrain-relative-navigation-hirise-dtm-mosaic)
- [USGS Jezero / MSR landing site products](https://astrogeology.usgs.gov/search/map/mars-2020-jezero-crater-landing-site-products)
- [NASA: Terrain Relative Navigation](https://science.nasa.gov/technology/technology-highlights/terrain-relative-navigation-landing-between-the-hazards)
- [USGS SIM 3464 geologic map of Jezero and Nili Planum](https://pubs.usgs.gov/publication/sim3464)
- [Mars 2020 ellipse mineralogy map (Zenodo)](https://zenodo.org/record/5575824)
- [Robbins V1 crater database (USGS pygeoapi)](https://astrogeology.usgs.gov/pygeoapi/collections/mars/robbinsv1)
- [USGS Murray Lab global CTX mosaic](https://astrogeology.usgs.gov/search/map/mars-mro-ctx-global-mosaic-murray-lab-v1)
- [CTX DTM mosaics over Gale (Zenodo)](https://zenodo.org/record/5808356)
- [NASA PIA22232: InSight landing site](https://pds-rings.seti.org/press_releases/pages/PIA22xxx/PIA22232.html)
