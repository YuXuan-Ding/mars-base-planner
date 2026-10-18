"""Fetch a coarse elevation grid of the land around the Jezero box, for the map's backdrop.

Same USGS HRSC+MOLA 200 m global DEM as fetch_dem.py, but covering 5x the box on each side
and sampled sparsely: one source row per output row, columns averaged in blocks.
It is deliberately low quality; only the inner box is used for planning.
Output: data/context_dem.i16 (int16 meters, row-major, north row first) + .json metadata.
"""
import json, struct, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
RES = 0.003374120830641          # degrees per pixel
WIDTH = 106694                   # global raster width
STRIP_OFFSETS_AT = 427148        # file offset of the strip-offset table (uint64, one strip per row)
NODATA = -32768

ROOT = Path(__file__).resolve().parent.parent
box = json.loads((ROOT / "data/jezero_dem_200m.json").read_text())
SCALE = 5                        # backdrop spans SCALE x the planning box
CN = 275                         # output grid is CN x CN (~4x coarser than the web map's cells)
clat = (box["lat_north"] + box["lat_south"]) / 2
clon = (box["lon_west"] + box["lon_east"]) / 2
hlat = (box["lat_north"] - box["lat_south"]) * SCALE / 2
hlon = (box["lon_east"] - box["lon_west"]) * SCALE / 2
LAT_N, LAT_S, LON_W, LON_E = clat + hlat, clat - hlat, clon - hlon, clon + hlon

def rng(a, n):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{a + n - 1}"})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            if attempt == 3:
                raise

r0, r1 = round((90 - LAT_N) / RES), round((90 - LAT_S) / RES)
c0, c1 = round((180 + LON_W) / RES), round((180 + LON_E) / RES)
cols = c1 - c0

def fetch(j):
    r = r0 + round((j + 0.5) * (r1 - r0) / CN)          # centre row of output row j
    off = struct.unpack("<Q", rng(STRIP_OFFSETS_AT + 8 * r, 8))[0]
    vals = struct.unpack(f"<{cols}h", rng(off + 2 * c0, 2 * cols))
    out = []
    for i in range(CN):
        blk = [v for v in vals[round(i * cols / CN):round((i + 1) * cols / CN)] if v != NODATA]
        out.append(round(sum(blk) / len(blk)) if blk else NODATA)
    return out

with ThreadPoolExecutor(32) as ex:
    grid = [v for row in ex.map(fetch, range(CN)) for v in row]

(ROOT / "data/context_dem.i16").write_bytes(struct.pack(f"<{len(grid)}h", *grid))
good = [v for v in grid if v != NODATA]
meta = {
    "source": "USGS Mars HRSC MOLA Blended DEM Global 200m v2, sparsely sampled",
    "url": URL, "units": "meters", "dtype": "int16 little-endian", "nodata": NODATA,
    "rows": CN, "cols": CN, "scale_of_box": SCALE,
    "lat_north": 90 - r0 * RES, "lat_south": 90 - r1 * RES,
    "lon_west": -180 + c0 * RES, "lon_east": -180 + c1 * RES,
    "min_m": min(good), "max_m": max(good), "nodata_count": len(grid) - len(good),
}
(ROOT / "data/context_dem.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
