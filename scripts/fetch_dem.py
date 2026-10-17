"""Crop the USGS HRSC+MOLA blended Mars DEM (200 m/px) to the Jezero planning box.

Reads only the needed bytes from the 11 GB global file with HTTP range requests.
Output: data/jezero_dem_200m.i16 (int16 meters, row-major, north row first) + .json metadata.
"""
import json, struct, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
RES = 0.003374120830641          # degrees per pixel
WIDTH = 106694                   # global raster width
STRIP_OFFSETS_AT = 427148        # file offset of the strip-offset table (uint64)
LAT_N, LAT_S, LON_W, LON_E = 21.9, 15.1, 73.4, 80.6

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
rows, cols = r1 - r0, c1 - c0
first = struct.unpack("<Q", rng(STRIP_OFFSETS_AT + 8 * r0, 8))[0]
rowbytes = WIDTH * 2

def fetch(i):
    return rng(first + i * rowbytes + 2 * c0, 2 * cols)

with ThreadPoolExecutor(32) as ex:
    data = b"".join(ex.map(fetch, range(rows)))

out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)
(out / "jezero_dem_200m.i16").write_bytes(data)
vals = struct.unpack(f"<{rows * cols}h", data)
good = [v for v in vals if v != -32768]
meta = {
    "source": "USGS Mars HRSC MOLA Blended DEM Global 200m v2",
    "url": URL, "units": "meters", "dtype": "int16 little-endian", "nodata": -32768,
    "rows": rows, "cols": cols, "deg_per_px": RES,
    "lat_north": 90 - r0 * RES, "lat_south": 90 - r1 * RES,
    "lon_west": -180 + c0 * RES, "lon_east": -180 + c1 * RES,
    "min_m": min(good), "max_m": max(good), "nodata_count": len(vals) - len(good),
}
(out / "jezero_dem_200m.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
