"""Fetch whole-Mars elevation for the zoomed-out globe view.

MOLA MEGDR, 4 pixels per degree (~15 km): 1440 x 720 big-endian int16 metres above the areoid,
north row first, columns from 0 deg E eastward. Same datum as the HRSC+MOLA blend used for the box.
Output: web/mars_globe.bin (int16 little-endian, same layout) + data/mars_globe.json metadata.
"""
import json, struct, urllib.request
from pathlib import Path

URL = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg004/megt90n000cb.img"
COLS, ROWS = 1440, 720
ROOT = Path(__file__).resolve().parent.parent

raw = urllib.request.urlopen(URL, timeout=120).read()
assert len(raw) == COLS * ROWS * 2, len(raw)
vals = struct.unpack(f">{COLS * ROWS}h", raw)
(ROOT / "web/mars_globe.bin").write_bytes(struct.pack(f"<{COLS * ROWS}h", *vals))
meta = {
    "source": "MGS MOLA MEGDR 4 px/deg (MEGT90N000CB), median topography above the areoid",
    "url": URL, "units": "meters", "dtype": "int16 little-endian", "rows": ROWS, "cols": COLS,
    "lat_north": 90, "lat_south": -90, "lon_west": 0, "lon_east": 360,
    "min_m": min(vals), "max_m": max(vals),
}
(ROOT / "data/mars_globe.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
