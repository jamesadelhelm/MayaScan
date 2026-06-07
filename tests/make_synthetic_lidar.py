#!/usr/bin/env python3
"""
Generate a synthetic LAS point cloud with Gaussian mounds at known positions.

Usage: python tests/make_synthetic_lidar.py
Outputs:
  data/lidar/synthetic_mounds.laz    -- point cloud (all ground, class 2)
  data/synthetic_mounds_truth.csv    -- name,lat,lon truth table
"""
import math
import pathlib
import numpy as np
import laspy
import csv

# ── Parameters ───────────────────────────────────────────────────────────────
SEED = 42
GRID_SPACING = 0.5        # m  (4 pts/m² per axis → 4 pts/m²)
NOISE_Z = 0.03            # m  background terrain noise

TILE_X0 = 282000.0        # UTM 16N easting origin (matches Caracol sample tile)
TILE_Y0 = 1880500.0       # UTM 16N northing origin
TILE_W = 500.0            # m  (500 × 500 m tile for a fast test)
TILE_H = 500.0

# Mounds to embed: (offset_e_m, offset_n_m, amplitude_m, sigma_m)
MOUNDS = [
    (80,  90,  0.55, 4.0),   # M1  dome-like
    (170, 140, 0.42, 5.5),   # M2  wider
    (260, 80,  0.38, 3.5),   # M3  small
    (90,  230, 0.61, 4.5),   # M4  prominent
    (210, 220, 0.35, 3.0),   # M5  small
    (310, 170, 0.50, 5.0),   # M6
    (140, 350, 0.45, 4.0),   # M7
    (370, 280, 0.58, 6.0),   # M8  larger footprint
    (60,  420, 0.40, 3.5),   # M9
    (280, 400, 0.33, 3.0),   # M10 just above threshold
    (420, 100, 0.62, 4.5),   # M11
    (450, 350, 0.47, 5.0),   # M12
    (350, 440, 0.36, 3.5),   # M13
    (200, 460, 0.55, 4.0),   # M14
    (470, 450, 0.41, 4.0),   # M15
]

# False-positive traps: elongated ridges (not valid mound candidates)
RIDGES = [
    # (x_center, y_center, amplitude, sigma_across, length)
    (130, 310, 0.40, 2.0, 60),   # ridge 1 (long east–west)
    (380, 180, 0.35, 2.5, 80),   # ridge 2
]


def gaussian(x, y, cx, cy, amp, sig):
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))


def build_terrain(xs, ys, rng):
    z = rng.normal(0, NOISE_Z, size=xs.shape)

    for ex, ey, amp, sig in MOUNDS:
        z += gaussian(xs, ys, TILE_X0 + ex, TILE_Y0 + ey, amp, sig)

    for cx, cy, amp, sigma_x, length in RIDGES:
        rx = xs - (TILE_X0 + cx)
        ry = ys - (TILE_Y0 + cy)
        # elongated along X; narrow in Y
        z += amp * np.exp(-(rx ** 2) / (2 * (length / 2) ** 2) - ry ** 2 / (2 * sigma_x ** 2))

    return z


def main():
    rng = np.random.default_rng(SEED)

    # ── Grid ─────────────────────────────────────────────────────────────────
    e_vals = np.arange(TILE_X0, TILE_X0 + TILE_W, GRID_SPACING)
    n_vals = np.arange(TILE_Y0, TILE_Y0 + TILE_H, GRID_SPACING)
    ee, nn = np.meshgrid(e_vals, n_vals)
    xs = ee.ravel()
    ys = nn.ravel()
    zs = build_terrain(xs, ys, rng)

    # Add a second sparser random layer (simulates returns from low shrubs)
    n_extra = len(xs) // 5
    xe = rng.uniform(TILE_X0, TILE_X0 + TILE_W, n_extra)
    ye = rng.uniform(TILE_Y0, TILE_Y0 + TILE_H, n_extra)
    ze = build_terrain(xe, ye, rng) + rng.uniform(0.05, 1.5, n_extra)  # vegetation above

    all_x = np.concatenate([xs, xe])
    all_y = np.concatenate([ys, ye])
    all_z = np.concatenate([zs, ze])
    # Class 2 = ground for grid; class 1 = unclassified for extras
    classification = np.concatenate([
        np.full(len(xs), 2, dtype=np.uint8),
        np.full(len(xe), 1, dtype=np.uint8),
    ])

    # ── Write LAS ─────────────────────────────────────────────────────────────
    out_path = pathlib.Path("data/lidar/synthetic_mounds.las")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = np.array([TILE_X0, TILE_Y0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])

    tmp_path = out_path.with_suffix(".tmp.las")
    las = laspy.LasData(header=header)
    las.x = all_x
    las.y = all_y
    las.z = all_z
    las.classification = classification
    las.write(str(tmp_path))

    # Use PDAL to embed the UTM 16N CRS so MayaScan can export WGS84 coordinates
    import json, subprocess, tempfile
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(tmp_path), "override_srs": "EPSG:32616"},
            {"type": "writers.las", "filename": str(out_path), "a_srs": "EPSG:32616"},
        ]
    }
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        json.dump(pipeline, tf)
        tf_path = tf.name
    subprocess.run(["pdal", "pipeline", tf_path], check=True)
    tmp_path.unlink(missing_ok=True)
    pathlib.Path(tf_path).unlink(missing_ok=True)

    print(f"Wrote {len(all_x):,} points to {out_path}")
    print(f"  Grid: {len(xs):,} ground pts  +  {len(xe):,} unclassified")

    # ── Write truth CSV ───────────────────────────────────────────────────────
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)

    truth_path = pathlib.Path("data/synthetic_mounds_truth.csv")
    with truth_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "lat", "lon", "amplitude_m", "sigma_m"])
        for i, (de, dn, amp, sig) in enumerate(MOUNDS, 1):
            lon, lat = t.transform(TILE_X0 + de, TILE_Y0 + dn)
            w.writerow([f"M{i}", f"{lat:.8f}", f"{lon:.8f}", amp, sig])
            print(f"  M{i}: ({TILE_X0+de:.0f},{TILE_Y0+dn:.0f}) → ({lat:.6f},{lon:.6f})")

    print(f"\nWrote truth table ({len(MOUNDS)} mounds) to {truth_path}")
    print(f"Ridge traps: {len(RIDGES)} (should NOT be detected)")


if __name__ == "__main__":
    main()
