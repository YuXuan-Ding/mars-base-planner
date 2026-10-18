"""Build the web map's terrain grid from the 200 m DEM crop, Robbins craters and the M20 traverse.

Output: web/terrain.js defining window.TERRAIN (base64-packed per-cell arrays + vector data).
Pure standard library so it runs anywhere.
"""
import array, base64, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = 480                                  # particle grid is N x N cells (~0.84 km each)
MARS_R = 3396190.0
DEG_M = 2 * math.pi * MARS_R / 360

meta = json.loads((ROOT / "data/jezero_dem_200m.json").read_text())
R, C, RES = meta["rows"], meta["cols"], meta["deg_per_px"]
dem = array.array("h"); dem.frombytes((ROOT / "data/jezero_dem_200m.i16").read_bytes())
assert len(dem) == R * C

# per-pixel slope (degrees), central differences, true metric spacing at each latitude
dy = RES * DEG_M
slope = array.array("f", bytes(4 * R * C))
for r in range(1, R - 1):
    lat = meta["lat_north"] - (r + 0.5) * RES
    dx = RES * DEG_M * math.cos(math.radians(lat))
    base = r * C
    up, dn = base - C, base + C
    for c in range(1, C - 1):
        gx = (dem[base + c + 1] - dem[base + c - 1]) / (2 * dx)
        gy = (dem[up + c] - dem[dn + c]) / (2 * dy)
        slope[base + c] = math.degrees(math.atan(math.hypot(gx, gy)))

H, S, SM, RO = [], [], [], []
for j in range(N):                       # j = 0 is the southern row of cells
    r_hi = R - round(j * R / N); r_lo = R - round((j + 1) * R / N)
    for i in range(N):
        c_lo = round(i * C / N); c_hi = round((i + 1) * C / N)
        hs = ss = ss2 = 0.0; smax = 0.0; n = 0
        for r in range(max(r_lo, 1), min(r_hi, R - 1)):
            b = r * C
            for c in range(max(c_lo, 1), min(c_hi, C - 1)):
                s = slope[b + c]; hs += dem[b + c]; ss += s; ss2 += s * s; n += 1
                if s > smax: smax = s
        mean_s = ss / n
        H.append(round(hs / n)); S.append(mean_s); SM.append(smax)
        RO.append(math.sqrt(max(0.0, ss2 / n - mean_s * mean_s)))

def pack(values, typecode, scale=1.0, hi=None):
    a = array.array(typecode, [min(hi, round(v * scale)) if hi else round(v * scale) for v in values])
    return base64.b64encode(a.tobytes()).decode()

# craters inside the box
u_of = lambda lon: (lon - meta["lon_west"]) / (meta["lon_east"] - meta["lon_west"])
v_of = lambda lat: (lat - meta["lat_south"]) / (meta["lat_north"] - meta["lat_south"])
cr = json.loads((ROOT / "data/raw/robbins_jezero.json").read_text())["features"]
craters = sorted(([round(f["properties"]["lon_e"], 4), round(f["properties"]["lat"], 4), f["properties"]["diamkm"]] for f in cr), key=lambda x: -x[2])

# traverse, decimated to ~40 m spacing
tr = json.loads((ROOT / "data/raw/M20_traverse.json").read_text())
pts = []
for f in tr["features"]:
    g = f["geometry"]; lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
    for line in lines:
        for lon, lat, *_ in line:
            if pts:
                plon, plat = pts[-1]
                d = math.hypot((lon - plon) * math.cos(math.radians(lat)), lat - plat) * DEG_M
                if d < 40: continue
            pts.append((round(lon, 5), round(lat, 5)))
last_sol = max(f["properties"].get("sol", 0) for f in tr["features"])

out = {
    "N": N, "meta": meta, "units": {"H": "m int16", "S": "deg*10 uint16", "SM": "deg*10 uint16", "RO": "deg*10 uint16"},
    "H": pack(H, "h"), "S": pack(S, "H", 10), "SM": pack(SM, "H", 10), "RO": pack(RO, "H", 10),
    "craters": craters, "traverse": pts, "traverse_last_sol": last_sol,
}

# coarse backdrop around the box (scripts/fetch_context.py), flipped so row 0 is the south edge like H
ctx_path = ROOT / "data/context_dem.i16"
if ctx_path.exists():
    cmeta = json.loads((ROOT / "data/context_dem.json").read_text())
    cdem = array.array("h"); cdem.frombytes(ctx_path.read_bytes())
    cn = cmeta["cols"]
    rows = [cdem[r * cn:(r + 1) * cn] for r in range(cmeta["rows"])][::-1]
    out["context"] = {"N": cn, "meta": cmeta, "H": pack([v for row in rows for v in row], "h")}
(ROOT / "web").mkdir(exist_ok=True)
(ROOT / "web/terrain.js").write_text("window.TERRAIN = " + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"cells {N}x{N}, H {min(H)}..{max(H)} m, mean slope max {max(S):.1f}°, pixel slope max {max(SM):.1f}°")
print(f"craters {len(craters)} (largest {craters[0]}), traverse pts {len(pts)} to sol {last_sol}")
print("terrain.js bytes", (ROOT / "web/terrain.js").stat().st_size)
