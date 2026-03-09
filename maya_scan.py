#!/usr/bin/env python3
"""
maya_scan.py

One-file LiDAR terrain-anomaly pipeline:
- LAZ/LAS -> DTM (PDAL) GeoTIFF
- DTM -> multi-scale Local Relief Model (LRM)
- LRM -> candidate regions (connected components + terrain filters)
- optional multi-threshold consensus filtering with overlap-aware region matching
- candidates -> density raster + scores
- candidates -> DBSCAN clustering in meters (auto eps supported; auto-UTM if needed)
- exports: CSV, GeoJSON, KML, Markdown, HTML, and optional PDF
- candidate exports include cluster_id and dist_to_core_km

Project-goal improvements:
- region shape metrics: extent, aspect ratio, width/height, compactness, solidity
- post-filters for peak, area, slope, spacing, prominence, compactness, and solidity
- extra diagnostic plots and safer output overwrite behavior

Dependencies:
- Required: PDAL, numpy, scipy, rasterio, pyproj, matplotlib
- Optional: scikit-learn (DBSCAN), reportlab (PDF)

Responsible use:
- Output coordinates can correspond to sensitive locations. Handle responsibly.

Example:
python maya_scan.py \
  -i data/lidar/bz_hr_las31_crs.laz \
  --name bz_hr_tile_31_v2_shape_filters \
  --overwrite \
  --try-smrf \
  --pos-thresh auto:p96 \
  --min-density auto:p60 \
  --density-sigma 40 \
  --min-peak 0.50 \
  --min-area-m2 25 \
  --min-extent 0.38 \
  --max-aspect 3.5 \
  --edge-buffer-m 10 \
  --min-spacing-m 15 \
  --min-prominence 0.10 \
  --max-slope-deg 20 \
  --min-compactness 0.12 \
  --min-solidity 0.50 \
  --cluster-eps auto \
  --min-samples 4 \
  --report-top-n 30 \
  --label-top-n 60
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import time
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import xy as pix2map_xy
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening, gaussian_filter, label as cc_label
from scipy.spatial import ConvexHull, QhullError
from pyproj import CRS, Transformer

# Optional deps
try:
    from sklearn.cluster import DBSCAN  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:
    DBSCAN = None  # noqa
    NearestNeighbors = None  # noqa

try:
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
except Exception:
    canvas = None  # noqa

LOG = logging.getLogger("maya_scan")


# -----------------------------
# Logging
# -----------------------------
def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "process.log"

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    LOG.addHandler(ch)
    LOG.addHandler(fh)
    LOG.info("Logging to %s", log_path)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = (proc.stdout or "").strip()
    if out:
        LOG.info(out)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


# -----------------------------
# Parameters / defaults
# -----------------------------
@dataclass
class Params:
    dtm_resolution_m: float = 1.0

    # Multi-scale LRM sigmas (pixels)
    lrm_sigmas_small: Tuple[float, ...] = (1.0, 2.0)
    lrm_sigmas_large: Tuple[float, ...] = (8.0, 12.0, 16.0)

    # Candidate detection
    pos_relief_threshold_spec: str = "auto:p96"
    consensus_enabled: bool = True
    consensus_percentiles: Tuple[float, ...] = (95.0, 96.0, 97.0)
    consensus_min_support: int = 2
    consensus_match_radius_m: float = 12.0
    min_region_pixels: int = 20
    morph_open_iters: int = 1
    morph_close_iters: int = 1
    # Region-level terrain filter: drop regions whose slope q75 exceeds this.
    max_slope_deg: float = 20.0

    # Density
    density_sigma_pix: float = 40.0
    min_density_spec: str = "auto:p60"

    # Post-filters (shape/physics)
    min_peak_relief_m: float = 0.50        # e.g. 0.4 to cut noise
    min_area_m2: float = 25.0              # e.g. 30
    max_area_m2: float = 1200.0            # upper bound to suppress very large terrain blobs (<=0 disables)
    min_extent: float = 0.38               # bbox fill, 0..1 (e.g. 0.35)
    max_aspect: float = 3.5                # width/height or height/width (e.g. 4.0)
    edge_buffer_m: float = 10.0            # drop regions touching tile edge within this distance
    min_candidate_spacing_m: float = 15.0  # score-ordered de-dup spacing between candidate centers
    prominence_ring_pixels: int = 6        # ring width (pixels) for local prominence estimate
    min_prominence_m: float = 0.10         # region mean relief - local ring mean relief
    min_compactness: float = 0.12          # 4*pi*A/P^2 in [0,1], lower = line-like
    min_solidity: float = 0.50             # A / convex_hull_A in [0,1], lower = fragmented/linear

    # Clustering
    cluster_eps_m: float = 150.0
    cluster_eps_mode: str = "auto"  # "auto" or "fixed"
    cluster_min_samples: int = 4

    # Outputs
    kml_label_top_n: int = 50
    report_top_n: int = 25

    # Cutouts / HTML
    html_report: bool = True
    cutout_size_m: float = 140.0  # square window width in meters
    cutout_dpi: int = 160

    # Score
    # score = density^a * peak_relief^b * extent^c * consensus_support^d * prominence^e * compactness^f * solidity^g * area_m2^h
    score_density_exp: float = 1.0
    score_peak_exp: float = 1.0
    score_extent_exp: float = 0.35
    score_consensus_exp: float = 0.40
    score_prominence_exp: float = 0.75
    score_compactness_exp: float = 0.20
    score_solidity_exp: float = 0.20
    score_area_exp: float = 0.50


_RUN_NAME_BAD_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_AUTO_PERCENTILE_RE = re.compile(r"^auto:p([0-9]+(?:\.[0-9]+)?)$", re.IGNORECASE)


def _parse_float_arg(raw: str, field: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{field} must be a number, got '{raw}'.") from exc


def _arg_positive_float(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if v <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return v


def _arg_nonnegative_float(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if v < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return v


def _arg_unit_interval(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return v


def _arg_ge_one_float(raw: str) -> float:
    v = _parse_float_arg(raw, "Value")
    if v < 1.0:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return v


def _arg_positive_int(raw: str) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Value must be an integer, got '{raw}'.") from exc
    if v < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return v


def _arg_nonnegative_int(raw: str) -> int:
    try:
        v = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Value must be an integer, got '{raw}'.") from exc
    if v < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return v


def _normalized_auto_percentile_spec(raw: str) -> Optional[str]:
    s = str(raw).strip().lower()
    m = _AUTO_PERCENTILE_RE.match(s)
    if not m:
        return None
    p = float(m.group(1))
    if not (0.0 <= p <= 100.0):
        raise argparse.ArgumentTypeError("Auto percentile must be between 0 and 100 (auto:pXX).")
    return f"auto:p{p:g}"


def _arg_pos_thresh_spec(raw: str) -> str:
    auto = _normalized_auto_percentile_spec(raw)
    if auto is not None:
        return auto
    _ = _parse_float_arg(str(raw).strip(), "pos-thresh")
    return str(raw).strip().lower()


def _arg_min_density_spec(raw: str) -> str:
    auto = _normalized_auto_percentile_spec(raw)
    if auto is not None:
        return auto
    v = _parse_float_arg(str(raw).strip(), "min-density")
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError("min-density numeric value must be between 0 and 1.")
    return str(raw).strip().lower()


def _arg_cluster_eps_spec(raw: str) -> str:
    s = str(raw).strip().lower()
    if s == "auto":
        return "auto"
    v = _parse_float_arg(s, "cluster-eps")
    if v <= 0:
        raise argparse.ArgumentTypeError("cluster-eps must be > 0 or 'auto'.")
    return f"{v:g}"


def _arg_percentiles_csv(raw: str) -> str:
    s = str(raw).strip()
    if not s:
        raise argparse.ArgumentTypeError("Percentile list must be non-empty (example: 95,96,97).")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Percentile list must contain comma-separated numbers.")
    vals: List[float] = []
    for p in parts:
        v = _parse_float_arg(p, "consensus-percentiles")
        if not (0.0 <= v <= 100.0):
            raise argparse.ArgumentTypeError("Consensus percentiles must be between 0 and 100.")
        vals.append(v)
    uniq = sorted({round(v, 6) for v in vals})
    return ",".join(f"{v:g}" for v in uniq)


def sanitize_run_name(raw: str) -> str:
    name = str(raw).strip()
    name = _RUN_NAME_BAD_CHARS_RE.sub("_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._-")
    if not name:
        raise ValueError("Run name is empty after sanitization.")
    if name in {".", ".."}:
        raise ValueError("Run name cannot be '.' or '..'.")
    return name[:120]


def ensure_path_within(parent: Path, child: Path) -> None:
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(
            f"Resolved output path is outside runs-dir.\n"
            f"runs-dir={parent}\n"
            f"out-dir={child}"
        ) from exc


# -----------------------------
# PDAL DTM builder
# -----------------------------
def pdal_version() -> str:
    try:
        proc = subprocess.run(["pdal", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return (proc.stdout or "").strip()
    except Exception:
        return "pdal (unknown)"


def build_dtm_from_laz(
    laz_path: Path,
    out_dtm_tif: Path,
    tmp_dir: Path,
    resolution_m: float,
    try_smrf: bool,
) -> None:
    """
    Robust PDAL approach:
    - optionally classify ground with SMRF into temp LAS
    - write DTM via writers.gdal using gdalopts
    """
    if not laz_path.exists():
        raise FileNotFoundError(
            f"Input LAZ/LAS not found: {laz_path}\n"
            f"Tip: pass the real tile path."
        )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_path = laz_path

    if try_smrf:
        smrf_out = tmp_dir / "ground_smrf.las"
        smrf_json = {
            "pipeline": [
                {"type": "readers.las", "filename": str(in_path)},
                {
                    "type": "filters.smrf",
                    "ignore": "Classification[7:7]",
                    "scalar": 1.25,
                    "slope": 0.15,
                    "threshold": 0.5,
                    "window": 16.0,
                },
                {"type": "writers.las", "filename": str(smrf_out)},
            ]
        }
        smrf_path = tmp_dir / "pdal_smrf.json"
        smrf_path.write_text(json.dumps(smrf_json, indent=2), encoding="utf-8")
        LOG.info("Running: pdal pipeline %s", smrf_path)
        run_cmd(["pdal", "pipeline", str(smrf_path)])
        in_path = smrf_out

    pipeline: List[Dict[str, Any]] = [{"type": "readers.las", "filename": str(in_path)}]
    if try_smrf:
        pipeline.append({"type": "filters.range", "limits": "Classification[2:2]"})

    pipeline.append(
        {
            "type": "writers.gdal",
            "filename": str(out_dtm_tif),
            "resolution": float(resolution_m),
            "output_type": "min",
            "dimension": "Z",
            "data_type": "float32",
            "nodata": -9999.0,
            "gdaldriver": "GTiff",
            "gdalopts": [
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "BIGTIFF=IF_SAFER",
            ],
        }
    )

    dtm_json = {"pipeline": pipeline}
    dtm_path = tmp_dir / "pdal_dtm.json"
    dtm_path.write_text(json.dumps(dtm_json, indent=2), encoding="utf-8")
    LOG.info("Running: pdal pipeline %s", dtm_path)
    run_cmd(["pdal", "pipeline", str(dtm_path)])

    if not out_dtm_tif.exists():
        raise RuntimeError(f"DTM did not get written (expected {out_dtm_tif})")


# -----------------------------
# Raster helpers
# -----------------------------
def load_raster(path: Path) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def _res_m_from_profile(profile: Dict[str, Any]) -> float:
    t = profile["transform"]
    resx = float(abs(t.a))
    resy = float(abs(t.e))
    return float((resx + resy) / 2.0)


def write_float_geotiff(path: Path, arr: np.ndarray, base_profile: Dict[str, Any]) -> None:
    prof = dict(base_profile)
    prof.update(
        dtype="float32",
        count=1,
        nodata=None,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def compute_slope_degrees(dtm: np.ndarray, res_m: float) -> np.ndarray:
    fill = np.nanmedian(dtm)
    dtm_f = np.where(np.isnan(dtm), fill, dtm)
    dz_dy, dz_dx = np.gradient(dtm_f, res_m, res_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return np.degrees(slope_rad).astype("float32")


def hillshade(dtm: np.ndarray, res_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    fill = np.nanmedian(dtm)
    z = np.where(np.isnan(dtm), fill, dtm).astype("float32")
    dy, dx = np.gradient(z, res_m, res_m)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1).astype("float32")


# -----------------------------
# LRM
# -----------------------------
def build_multiscale_lrm(dtm: np.ndarray, params: Params) -> np.ndarray:
    fill = np.nanmedian(dtm)
    dtm_f = np.where(np.isnan(dtm), fill, dtm).astype("float32")

    lrms: List[np.ndarray] = []
    for s_small in params.lrm_sigmas_small:
        small = gaussian_filter(dtm_f, sigma=s_small)
        for s_large in params.lrm_sigmas_large:
            if s_large <= s_small:
                continue
            large = gaussian_filter(dtm_f, sigma=s_large)
            lrms.append(small - large)

    if not lrms:
        raise RuntimeError("No valid sigma pairs for LRM")
    return np.maximum.reduce(lrms).astype("float32")


def parse_auto_percentile(spec: str, values: np.ndarray, positive_only: bool = True) -> float:
    spec_norm = str(spec).strip().lower()
    if not spec_norm.startswith("auto:p"):
        return float(spec_norm)

    m = _AUTO_PERCENTILE_RE.match(spec_norm)
    if not m:
        raise ValueError(f"Invalid auto percentile spec: '{spec}'. Expected auto:pXX.")
    p = float(m.group(1))
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"Auto percentile must be between 0 and 100, got {p}.")

    vals = values[np.isfinite(values)]
    if positive_only:
        vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, p))


# -----------------------------
# Candidates
# -----------------------------
@dataclass
class Candidate:
    cand_id: int
    px_x: float
    px_y: float
    peak_relief_m: float
    mean_relief_m: float
    area_m2: float
    density: float
    extent: float              # bbox fill (0..1)
    aspect: float              # >=1
    consensus_support: int     # number of threshold runs that support this region
    prominence_m: float        # mean relief over region minus local ring mean
    compactness: float         # 4*pi*A/P^2 in [0,1]
    solidity: float            # A / convex_hull_A in [0,1]
    width_m: float
    height_m: float
    score: float
    lon: float
    lat: float
    cluster_id: int = -1
    dist_to_core_km: Optional[float] = None  # distance to densest candidate within the same cluster
    img_relpath: Optional[str] = None


def _region_perimeter_pixels(region_mask: np.ndarray) -> float:
    """
    Estimate perimeter from exposed unit-length cell edges.
    This is more stable than counting boundary pixels and avoids
    artificially inflating compactness for small regions.
    """
    if region_mask.size == 0:
        return 0.0
    padded = np.pad(region_mask.astype(bool), 1, constant_values=False)
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return float(vertical + horizontal)


def _region_solidity(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Solidity = region area / convex hull area in raster pixel units.
    Use pixel corners rather than pixel centers so the hull area reflects
    the occupied raster cells instead of the center-point cloud.
    Values near 1 are compact/filled; lower values are fragmented/linear.
    """
    n = int(xs.size)
    if n < 3:
        return 1.0
    xs64 = xs.astype("float64")
    ys64 = ys.astype("float64")
    pts = np.concatenate(
        [
            np.column_stack([xs64, ys64]),
            np.column_stack([xs64 + 1.0, ys64]),
            np.column_stack([xs64, ys64 + 1.0]),
            np.column_stack([xs64 + 1.0, ys64 + 1.0]),
        ],
        axis=0,
    )
    pts = np.unique(pts, axis=0)
    try:
        hull = ConvexHull(pts)
        hull_area_pix2 = float(hull.volume)  # 2D hull area
    except (QhullError, ValueError):
        return 1.0
    if hull_area_pix2 <= 1e-9:
        return 1.0
    return float(np.clip(float(n) / hull_area_pix2, 0.0, 1.0))


def detect_regions(
    lrm: np.ndarray,
    dtm_slope_deg: np.ndarray,
    profile: Dict[str, Any],
    params: Params,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Extract connected components above pos_relief threshold.
    Compute shape metrics per region.
    Returns regions list + pos_thresh used.
    """
    _, regions, _, pos_thresh = _extract_candidate_regions(lrm, dtm_slope_deg, profile, params)
    return regions, pos_thresh


def _consensus_specs_from_params(params: Params) -> List[str]:
    base_spec = str(params.pos_relief_threshold_spec).strip().lower()
    if not params.consensus_enabled:
        return [base_spec]

    base_auto = _normalized_auto_percentile_spec(base_spec)
    if base_auto is None:
        return [base_spec]

    vals: List[float] = []
    for raw in params.consensus_percentiles:
        try:
            p = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 <= p <= 100.0:
            vals.append(p)

    if not vals:
        vals = [95.0, 96.0, 97.0]

    base_p = float(base_auto.split("auto:p", 1)[1])
    vals.append(base_p)
    uniq = sorted({round(v, 6) for v in vals})
    return [f"auto:p{v:g}" for v in uniq]


def _attach_region_density_stats(regions: List[Dict[str, Any]], labeled: np.ndarray, density_norm: np.ndarray) -> None:
    for r in regions:
        rid = int(r["rid"])
        dens_vals = density_norm[labeled == rid]
        dens_vals = dens_vals[np.isfinite(dens_vals)]
        if dens_vals.size == 0:
            r["density_mean"] = 0.0
            r["density_q75"] = 0.0
            continue
        r["density_mean"] = float(np.mean(dens_vals))
        r["density_q75"] = float(np.percentile(dens_vals, 75))


def _compute_primary_consensus_support(
    primary_labeled: np.ndarray,
    primary_regions: List[Dict[str, Any]],
    other_runs: List[Tuple[np.ndarray, List[Dict[str, Any]]]],
    match_radius_pix: float,
) -> None:
    if not primary_regions:
        return

    r2 = float(max(0.0, match_radius_pix * match_radius_pix))
    pad = int(math.ceil(max(0.0, match_radius_pix)))
    other_region_maps = [{int(r["rid"]): r for r in regs} for _, regs in other_runs]

    for r in primary_regions:
        rid = int(r["rid"])
        x0 = max(0, int(r["x0"]) - pad)
        y0 = max(0, int(r["y0"]) - pad)
        x1 = min(primary_labeled.shape[1] - 1, int(r["x1"]) + pad)
        y1 = min(primary_labeled.shape[0] - 1, int(r["y1"]) + pad)
        primary_mask = primary_labeled[y0 : y1 + 1, x0 : x1 + 1] == rid
        primary_pix = int(primary_mask.sum())
        cx = float(r["cx"])
        cy = float(r["cy"])
        support = 1

        for (other_labeled, _), other_region_map in zip(other_runs, other_region_maps):
            window = other_labeled[y0 : y1 + 1, x0 : x1 + 1]
            cand_rids = np.unique(window[window > 0])
            matched = False

            for other_rid in cand_rids:
                other_region = other_region_map.get(int(other_rid))
                if other_region is None:
                    continue

                other_mask = window == int(other_rid)
                overlap_pix = int(np.count_nonzero(primary_mask & other_mask))
                if overlap_pix <= 0:
                    continue

                other_pix = max(1, int(other_region["pixels"]))
                overlap_frac = float(overlap_pix) / float(min(primary_pix, other_pix))
                d2 = (float(other_region["cx"]) - cx) ** 2 + (float(other_region["cy"]) - cy) ** 2

                # Require real raster overlap; center distance only guards against
                # large merged regions being treated as a match too loosely.
                if overlap_frac >= 0.05 and (d2 <= r2 or overlap_frac >= 0.25):
                    matched = True
                    break

            if matched:
                support += 1
        r["consensus_support"] = int(support)


def _extract_candidate_regions(
    lrm: np.ndarray,
    dtm_slope_deg: np.ndarray,
    profile: Dict[str, Any],
    params: Params,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[int], float]:
    res_m = _res_m_from_profile(profile)

    pos_thresh = parse_auto_percentile(params.pos_relief_threshold_spec, lrm, positive_only=True)
    LOG.info("Positive relief threshold (m): %.4f (spec=%s)", pos_thresh, params.pos_relief_threshold_spec)

    lrm_filled = np.where(np.isfinite(lrm), lrm, 0.0).astype("float32")
    mask = lrm_filled > float(pos_thresh)

    mask = binary_opening(mask, iterations=params.morph_open_iters)
    mask = binary_closing(mask, iterations=params.morph_close_iters)

    labeled, n = cc_label(mask)
    LOG.info("Initial regions: %d", n)

    regions: List[Dict[str, Any]] = []
    kept_rids: List[int] = []

    for rid in range(1, n + 1):
        region = labeled == rid
        pix = int(region.sum())
        if pix < params.min_region_pixels:
            continue

        ys, xs = np.where(region)
        cy = float(ys.mean())
        cx = float(xs.mean())

        slope_vals = dtm_slope_deg[region]
        slope_vals = slope_vals[np.isfinite(slope_vals)]
        if slope_vals.size == 0:
            continue
        slope_median_deg = float(np.percentile(slope_vals, 50))
        slope_q75_deg = float(np.percentile(slope_vals, 75))
        slope_max_deg = float(np.max(slope_vals))
        if slope_q75_deg > params.max_slope_deg:
            continue

        peak = float(np.nanmax(lrm[region]))
        mean = float(np.nanmean(lrm[region]))
        ring_iters = max(1, int(params.prominence_ring_pixels))
        ring_mask = binary_dilation(region, iterations=ring_iters) & ~region
        ring_vals = lrm[ring_mask]
        ring_vals = ring_vals[np.isfinite(ring_vals)]
        ring_mean_relief_m = float(np.mean(ring_vals)) if ring_vals.size else mean
        prominence_m = float(mean - ring_mean_relief_m)

        # bbox shape metrics
        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())
        w_pix = float((x1 - x0 + 1))
        h_pix = float((y1 - y0 + 1))
        w_m = w_pix * res_m
        h_m = h_pix * res_m

        bbox_area_m2 = max(1e-9, w_m * h_m)
        area_m2 = pix * res_m * res_m

        extent = float(np.clip(area_m2 / bbox_area_m2, 0.0, 1.0))
        aspect = float(max(w_m / max(1e-9, h_m), h_m / max(1e-9, w_m)))  # >=1
        perimeter_m = _region_perimeter_pixels(region) * res_m
        compactness = float(
            np.clip((4.0 * math.pi * area_m2) / max(1e-9, perimeter_m * perimeter_m), 0.0, 1.0)
        )
        solidity = _region_solidity(xs, ys)

        regions.append(
            {
                "rid": rid,
                "cx": cx,
                "cy": cy,
                "pixels": pix,
                "area_m2": area_m2,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "peak": peak,
                "mean": mean,
                "ring_mean_relief_m": ring_mean_relief_m,
                "prominence_m": prominence_m,
                "extent": extent,
                "aspect": aspect,
                "width_m": w_m,
                "height_m": h_m,
                "perimeter_m": perimeter_m,
                "compactness": compactness,
                "solidity": solidity,
                "slope_median_deg": slope_median_deg,
                "slope_q75_deg": slope_q75_deg,
                "slope_max_deg": slope_max_deg,
            }
        )
        kept_rids.append(rid)

    LOG.info("Filtered regions (after size + region-slope q75): %d", len(regions))
    return labeled, regions, kept_rids, pos_thresh


def build_density_from_regions(
    labeled_regions: np.ndarray,
    kept_rids: List[int],
    profile: Dict[str, Any],
    params: Params,
    out_density_tif: Path,
) -> np.ndarray:
    mound_binary = np.isin(labeled_regions, np.array(kept_rids, dtype=int)).astype("float32")
    density = gaussian_filter(mound_binary, sigma=float(params.density_sigma_pix)).astype("float32")
    dmin = float(np.min(density))
    dmax = float(np.max(density))
    density_norm = ((density - dmin) / (dmax - dmin + 1e-9)).astype("float32")

    write_float_geotiff(out_density_tif, density_norm, profile)
    LOG.info("Wrote density raster: %s", out_density_tif)
    return density_norm


def detect_candidates(
    lrm: np.ndarray,
    dtm_slope_deg: np.ndarray,
    profile: Dict[str, Any],
    params: Params,
    out_density_tif: Path,
) -> Tuple[List[Dict[str, Any]], np.ndarray, float, float, Dict[str, Any]]:
    """
    Candidate extraction wrapper with optional multi-threshold consensus support.
    Returns regions, density_norm, pos_thresh, min_density, diagnostics.
    """
    res_m = _res_m_from_profile(profile)
    specs = _consensus_specs_from_params(params)

    runs: List[Tuple[str, np.ndarray, List[Dict[str, Any]], List[int], float]] = []
    for spec in specs:
        p = replace(params, pos_relief_threshold_spec=spec)
        labeled_i, regions_i, kept_rids_i, pos_thresh_i = _extract_candidate_regions(lrm, dtm_slope_deg, profile, p)
        runs.append((spec, labeled_i, regions_i, kept_rids_i, pos_thresh_i))

    primary_idx = 0
    for i, (spec, _, _, _, _) in enumerate(runs):
        if spec == str(params.pos_relief_threshold_spec).strip().lower():
            primary_idx = i
            break

    primary_spec, labeled, regions, kept_rids, pos_thresh = runs[primary_idx]
    diagnostics: Dict[str, Any] = {
        "consensus_specs": [s for s, _, _, _, _ in runs],
        "consensus_regions_before": int(len(regions)),
        "consensus_dropped": 0,
        "consensus_enabled": bool(params.consensus_enabled and len(runs) > 1),
        "consensus_primary_spec": primary_spec,
    }

    if len(runs) > 1 and params.consensus_enabled and params.consensus_min_support > 1:
        radius_pix = float(max(0.0, params.consensus_match_radius_m) / max(1e-9, res_m))
        other_runs = [(labeled_i, regions_i) for i, (_, labeled_i, regions_i, _, _) in enumerate(runs) if i != primary_idx]
        _compute_primary_consensus_support(
            primary_labeled=labeled,
            primary_regions=regions,
            other_runs=other_runs,
            match_radius_pix=radius_pix,
        )
        before = len(regions)
        regions = [r for r in regions if int(r.get("consensus_support", 0)) >= int(params.consensus_min_support)]
        kept_rids = [int(r["rid"]) for r in regions]
        dropped = max(0, before - len(regions))
        diagnostics["consensus_dropped"] = int(dropped)
        diagnostics["consensus_regions_after"] = int(len(regions))
        diagnostics["consensus_radius_m"] = float(params.consensus_match_radius_m)
        diagnostics["consensus_min_support"] = int(params.consensus_min_support)
        LOG.info(
            "Consensus support filter: specs=%s | min_support=%d | radius=%.1fm | dropped=%d | kept=%d",
            ",".join(diagnostics["consensus_specs"]),
            int(params.consensus_min_support),
            float(params.consensus_match_radius_m),
            int(dropped),
            int(len(regions)),
        )
    else:
        for r in regions:
            r["consensus_support"] = 1
        diagnostics["consensus_regions_after"] = int(len(regions))
        if len(runs) <= 1:
            LOG.info("Consensus support filter disabled (single threshold spec=%s).", primary_spec)
        elif params.consensus_min_support <= 1:
            LOG.info("Consensus support filter disabled (consensus_min_support <= 1).")

    density_norm = build_density_from_regions(labeled, kept_rids, profile, params, out_density_tif)
    _attach_region_density_stats(regions, labeled, density_norm)

    min_density = parse_auto_percentile(params.min_density_spec, density_norm, positive_only=False)
    LOG.info("Min density threshold: %.4f (spec=%s)", min_density, params.min_density_spec)
    return regions, density_norm, pos_thresh, min_density, diagnostics


# -----------------------------
# Clustering (meters)
# -----------------------------
def _auto_dbscan_eps(coords_m: np.ndarray, min_samples: int) -> float:
    if NearestNeighbors is None or coords_m.shape[0] < max(10, min_samples + 1):
        return 150.0
    k = max(2, int(min_samples))
    nn = NearestNeighbors(n_neighbors=min(k + 1, coords_m.shape[0]))
    nn.fit(coords_m)
    dists, _ = nn.kneighbors(coords_m)
    kth = np.sort(dists[:, -1].astype("float64"))

    fallback = float(np.percentile(kth, 80))
    if kth.size < 4 or not np.all(np.isfinite(kth)) or float(kth[-1]) <= float(kth[0]):
        return float(np.clip(fallback, 40.0, 250.0))

    x = np.linspace(0.0, 1.0, kth.size)
    y = (kth - kth[0]) / max(1e-9, float(kth[-1] - kth[0]))
    knee_idx = int(np.argmax(y - x))
    eps = float(kth[knee_idx])
    if knee_idx <= 0 or knee_idx >= kth.size - 1:
        eps = fallback
    return float(np.clip(eps, 40.0, 250.0))


def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _projected_unit_factor_to_meters(src_crs: CRS) -> Optional[float]:
    factors: List[float] = []
    for axis in src_crs.axis_info:
        fac = getattr(axis, "unit_conversion_factor", None)
        if fac is None:
            continue
        try:
            val = float(fac)
        except (TypeError, ValueError):
            continue
        if val > 0:
            factors.append(val)

    if not factors:
        return None

    first = factors[0]
    for val in factors[1:]:
        if not math.isclose(first, val, rel_tol=1e-9, abs_tol=1e-12):
            LOG.warning(
                "Projected CRS has mixed axis conversion factors; using first factor %.12g.",
                first,
            )
            break
    return first


def project_points_to_meters(
    src_crs: CRS,
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, CRS]:
    """
    Ensure we have meter-based coords for clustering/distances.
    If src_crs is projected (meters-ish), pass through.
    If geographic (degrees), convert to EPSG:4326 then to UTM based on centroid.
    """
    xs64 = xs.astype("float64")
    ys64 = ys.astype("float64")

    if src_crs.is_projected:
        factor = _projected_unit_factor_to_meters(src_crs)
        if factor is not None:
            if math.isclose(factor, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                return xs64, ys64, src_crs
            LOG.info("Projected CRS units converted to meters (factor=%.12g).", factor)
            return xs64 * factor, ys64 * factor, src_crs
        LOG.warning("Projected CRS unit conversion factor unavailable; reprojecting to UTM for meter distances.")

    lon = xs64
    lat = ys64
    if src_crs.to_epsg() != 4326:
        to_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(lon, lat)

    clon = float(np.mean(lon))
    clat = float(np.mean(lat))
    utm_epsg = _utm_epsg_from_lonlat(clon, clat)
    utm_crs = CRS.from_epsg(utm_epsg)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x_m, y_m = to_utm.transform(lon, lat)
    return np.array(x_m, dtype="float64"), np.array(y_m, dtype="float64"), utm_crs


def dedupe_candidates_by_spacing(
    candidates: List[Candidate],
    src_crs: CRS,
    dtm_transform: Any,
    min_spacing_m: float,
) -> Tuple[List[Candidate], int]:
    """
    Score-ordered non-maximum suppression by center spacing in meters.
    Keeps highest-scoring candidate within each local neighborhood.
    """
    if min_spacing_m <= 0.0 or len(candidates) < 2:
        return candidates, 0

    xs_map: List[float] = []
    ys_map: List[float] = []
    for c in candidates:
        x_map, y_map = pix2map_xy(dtm_transform, c.px_y, c.px_x)
        xs_map.append(float(x_map))
        ys_map.append(float(y_map))

    xs = np.array(xs_map, dtype="float64")
    ys = np.array(ys_map, dtype="float64")
    xs_m, ys_m, _ = project_points_to_meters(src_crs, xs, ys)

    order = np.argsort(np.array([-c.score for c in candidates], dtype="float64"))
    keep_mask = np.zeros(len(candidates), dtype=bool)
    kept_idx: List[int] = []
    spacing2 = float(min_spacing_m * min_spacing_m)
    dropped = 0

    for idx in order:
        if kept_idx:
            d2 = (xs_m[kept_idx] - xs_m[idx]) ** 2 + (ys_m[kept_idx] - ys_m[idx]) ** 2
            if float(np.min(d2)) < spacing2:
                dropped += 1
                continue
        keep_mask[idx] = True
        kept_idx.append(int(idx))

    kept = [c for i, c in enumerate(candidates) if bool(keep_mask[i])]
    return kept, dropped


def cluster_candidates_meters(xs_m: np.ndarray, ys_m: np.ndarray, params: Params) -> np.ndarray:
    if DBSCAN is None:
        LOG.warning("sklearn not installed; clustering disabled (cluster_id=-1).")
        return np.full(xs_m.shape[0], -1, dtype=int)

    coords = np.column_stack([xs_m, ys_m]).astype("float32")

    eps = float(params.cluster_eps_m)
    if params.cluster_eps_mode == "auto":
        eps = _auto_dbscan_eps(coords, int(params.cluster_min_samples))
        LOG.info("DBSCAN eps auto-chosen: %.1f m (min_samples=%d)", eps, params.cluster_min_samples)
    else:
        LOG.info("DBSCAN eps fixed: %.1f m (min_samples=%d)", eps, params.cluster_min_samples)

    model = DBSCAN(eps=eps, min_samples=int(params.cluster_min_samples))
    labels = model.fit_predict(coords)

    out = labels.copy()
    next_id = 1
    mapping: Dict[int, int] = {}
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        mapping[lab] = next_id
        next_id += 1
    for i, lab in enumerate(labels):
        out[i] = mapping.get(lab, -1)
    return out.astype(int)


def assign_cluster_core_distances(candidates: List[Candidate], xs_m: np.ndarray, ys_m: np.ndarray) -> None:
    """
    Compute distance to the densest candidate within the same cluster.
    Noise candidates do not get a cluster-core distance.
    """
    if not candidates:
        return

    cluster_to_core_idx: Dict[int, int] = {}
    for idx, cand in enumerate(candidates):
        cid = int(cand.cluster_id)
        if cid == -1:
            continue
        cur = cluster_to_core_idx.get(cid)
        if cur is None:
            cluster_to_core_idx[cid] = idx
            continue
        if (cand.density, cand.score) > (candidates[cur].density, candidates[cur].score):
            cluster_to_core_idx[cid] = idx

    for idx, cand in enumerate(candidates):
        cid = int(cand.cluster_id)
        if cid == -1:
            cand.dist_to_core_km = None
            continue
        core_idx = cluster_to_core_idx.get(cid)
        if core_idx is None:
            cand.dist_to_core_km = None
            continue
        dx = float(xs_m[idx] - xs_m[core_idx])
        dy = float(ys_m[idx] - ys_m[core_idx])
        cand.dist_to_core_km = float(math.hypot(dx, dy) / 1000.0)


# -----------------------------
# Exporters
# -----------------------------
def write_geojson(candidates: List[Candidate], out_path: Path) -> None:
    feats = []
    for c in candidates:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c.lon, c.lat]},
                "properties": {
                    "cand_id": c.cand_id,
                    "score": c.score,
                    "density": c.density,
                    "peak_relief_m": c.peak_relief_m,
                    "mean_relief_m": c.mean_relief_m,
                    "area_m2": c.area_m2,
                    "extent": c.extent,
                    "aspect": c.aspect,
                    "consensus_support": c.consensus_support,
                    "prominence_m": c.prominence_m,
                    "compactness": c.compactness,
                    "solidity": c.solidity,
                    "width_m": c.width_m,
                    "height_m": c.height_m,
                    "cluster_id": c.cluster_id,
                    "dist_to_core_km": c.dist_to_core_km,
                },
            }
        )
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=2), encoding="utf-8")


def write_csv(candidates: List[Candidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cand_id",
                "score",
                "density",
                "peak_relief_m",
                "mean_relief_m",
                "area_m2",
                "extent",
                "aspect",
                "consensus_support",
                "prominence_m",
                "compactness",
                "solidity",
                "width_m",
                "height_m",
                "lon",
                "lat",
                "cluster_id",
                "dist_to_core_km",
            ]
        )
        for c in candidates:
            w.writerow(
                [
                    c.cand_id,
                    f"{c.score:.6f}",
                    f"{c.density:.6f}",
                    f"{c.peak_relief_m:.4f}",
                    f"{c.mean_relief_m:.4f}",
                    f"{c.area_m2:.2f}",
                    f"{c.extent:.4f}",
                    f"{c.aspect:.3f}",
                    c.consensus_support,
                    f"{c.prominence_m:.4f}",
                    f"{c.compactness:.4f}",
                    f"{c.solidity:.4f}",
                    f"{c.width_m:.2f}",
                    f"{c.height_m:.2f}",
                    f"{c.lon:.8f}",
                    f"{c.lat:.8f}",
                    c.cluster_id,
                    "" if c.dist_to_core_km is None else f"{c.dist_to_core_km:.4f}",
                ]
            )


def write_clusters_csv(candidates: List[Candidate], out_path: Path) -> None:
    clusters: Dict[int, List[Candidate]] = {}
    for c in candidates:
        if c.cluster_id == -1:
            continue
        clusters.setdefault(c.cluster_id, []).append(c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "n", "mean_score", "mean_density", "centroid_lon", "centroid_lat"])
        for cid in sorted(clusters.keys()):
            pts = clusters[cid]
            n = len(pts)
            mean_score = float(np.mean([p.score for p in pts])) if n else 0.0
            mean_dens = float(np.mean([p.density for p in pts])) if n else 0.0
            clon = float(np.mean([p.lon for p in pts])) if n else 0.0
            clat = float(np.mean([p.lat for p in pts])) if n else 0.0
            w.writerow([cid, n, f"{mean_score:.6f}", f"{mean_dens:.6f}", f"{clon:.8f}", f"{clat:.8f}"])


def _kml_escape(s: str) -> str:
    return s.replace("&", "and").replace("<", "(").replace(">", ")")


def write_kml(candidates: List[Candidate], out_path: Path, label_top_n: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = sorted_c[: max(0, int(label_top_n))]
    rest = sorted_c[max(0, int(label_top_n)) :]

    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>MayaScan Candidates</name>",
        '<Style id="topPin">',
        "<IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href></Icon></IconStyle>",
        "<LabelStyle><scale>1.0</scale></LabelStyle>",
        "</Style>",
        '<Style id="dot">',
        "<IconStyle><scale>0.35</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>",
        "<LabelStyle><scale>0</scale></LabelStyle>",
        "</Style>",
    ]

    def placemark(c: Candidate, labeled: bool, style: str) -> List[str]:
        desc = (
            f"cand_id={c.cand_id}<br/>"
            f"score={c.score:.4f}<br/>"
            f"density={c.density:.4f}<br/>"
            f"peak_relief_m={c.peak_relief_m:.3f}<br/>"
            f"area_m2={c.area_m2:.0f}<br/>"
            f"extent={c.extent:.3f}<br/>"
            f"aspect={c.aspect:.2f}<br/>"
            f"consensus_support={c.consensus_support}<br/>"
            f"prominence_m={c.prominence_m:.3f}<br/>"
            f"compactness={c.compactness:.3f}<br/>"
            f"solidity={c.solidity:.3f}<br/>"
            f"cluster_id={c.cluster_id}"
        )
        if c.dist_to_core_km is not None:
            desc += f"<br/>dist_to_core_km={c.dist_to_core_km:.3f}"

        name = f"Candidate {c.cand_id} (score={c.score:.2f})" if labeled else ""

        return [
            "<Placemark>",
            f"<name>{_kml_escape(name)}</name>",
            f"<styleUrl>#{style}</styleUrl>",
            f"<description>{desc}</description>",
            f"<Point><coordinates>{c.lon:.8f},{c.lat:.8f},0</coordinates></Point>",
            "</Placemark>",
        ]

    kml.append("<Folder><name>Top Ranked (labeled)</name>")
    for c in top:
        kml.extend(placemark(c, labeled=True, style="topPin"))
    kml.append("</Folder>")

    kml.append("<Folder><name>All Candidates (unlabeled)</name>")
    for c in rest:
        kml.extend(placemark(c, labeled=False, style="dot"))
    kml.append("</Folder>")

    kml.append("</Document></kml>")
    out_path.write_text("\n".join(kml), encoding="utf-8")


# -----------------------------
# Plots + report
# -----------------------------
def make_plots(out_dir: Path, lrm: np.ndarray, density: np.ndarray, candidates: List[Candidate]) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(density, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("Settlement Density (normalized)")
    plt.colorbar()
    p = plots_dir / "density.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote density PNG: %s", p)

    plt.figure(figsize=(10, 8))
    # robust clip for display only
    lrm_vals = lrm[np.isfinite(lrm)]
    if lrm_vals.size:
        lo, hi = np.percentile(lrm_vals, [2, 98])
    else:
        lo, hi = -1.0, 1.0
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        lo, hi = -1.0, 1.0
    lrm_show = np.clip(np.where(np.isfinite(lrm), lrm, 0.0), lo, hi)
    plt.imshow(lrm_show, cmap="gray", vmin=lo, vmax=hi)
    xs = [c.px_x for c in candidates]
    ys = [c.px_y for c in candidates]
    plt.scatter(xs, ys, s=12)
    plt.title("Candidates overlay on combined LRM")
    p = plots_dir / "candidates_overlay.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote overlay PNG: %s", p)

    scores = np.array([c.score for c in candidates], dtype=float)
    peaks = np.array([c.peak_relief_m for c in candidates], dtype=float)
    supports = np.array([c.consensus_support for c in candidates], dtype=float)
    prominence = np.array([c.prominence_m for c in candidates], dtype=float)
    areas = np.array([c.area_m2 for c in candidates], dtype=float)
    extents = np.array([c.extent for c in candidates], dtype=float)
    aspects = np.array([c.aspect for c in candidates], dtype=float)
    compactness = np.array([c.compactness for c in candidates], dtype=float)
    solidity = np.array([c.solidity for c in candidates], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.hist(scores[scores > 0], bins=30)
    plt.title("Score distribution (score>0)")
    plt.xlabel("score")
    plt.ylabel("count")
    p = plots_dir / "score_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(peaks, bins=30)
    plt.title("Peak relief distribution")
    plt.xlabel("peak relief (m)")
    plt.ylabel("count")
    p = plots_dir / "peak_relief_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(supports, bins=max(5, int(np.max(supports)) + 1 if supports.size else 5))
    plt.title("Consensus support distribution")
    plt.xlabel("support count")
    plt.ylabel("count")
    p = plots_dir / "consensus_support_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(prominence, bins=30)
    plt.title("Local prominence distribution")
    plt.xlabel("prominence (m)")
    plt.ylabel("count")
    p = plots_dir / "prominence_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(areas, bins=30)
    plt.title("Area distribution")
    plt.xlabel("area (m^2)")
    plt.ylabel("count")
    p = plots_dir / "area_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(extents, bins=30, range=(0, 1))
    plt.title("Extent distribution (bbox fill)")
    plt.xlabel("extent (0..1)")
    plt.ylabel("count")
    p = plots_dir / "extent_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(aspects, bins=30)
    plt.title("Aspect ratio distribution (>=1)")
    plt.xlabel("aspect")
    plt.ylabel("count")
    p = plots_dir / "aspect_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(compactness, bins=30, range=(0, 1))
    plt.title("Compactness distribution (4*pi*A/P^2)")
    plt.xlabel("compactness (0..1)")
    plt.ylabel("count")
    p = plots_dir / "compactness_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)

    plt.figure(figsize=(10, 5))
    plt.hist(solidity, bins=30, range=(0, 1))
    plt.title("Solidity distribution (area / convex_hull_area)")
    plt.xlabel("solidity (0..1)")
    plt.ylabel("count")
    p = plots_dir / "solidity_hist.png"
    plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()
    LOG.info("Wrote plot: %s", p)


def write_report_md(
    out_dir: Path,
    run_name: str,
    input_path: Path,
    dtm_path: Path,
    lrm_path: Path,
    density_path: Path,
    candidates: List[Candidate],
    clusters_csv: Path,
    params: Params,
    pos_thresh: float,
    min_density: float,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = sorted_c[: params.report_top_n]
    n_clusters = len({c.cluster_id for c in candidates if c.cluster_id != -1})
    n_noise = sum(1 for c in candidates if c.cluster_id == -1)

    md: List[str] = []
    md.append(f"# MayaScan report: {run_name}")
    md.append("")
    md.append(f"- Timestamp: **{ts}**")
    md.append(f"- Input: `{input_path}`")
    md.append("")
    md.append("## Outputs")
    md.append(f"- DTM: `{dtm_path}`")
    md.append(f"- LRM: `{lrm_path}`")
    md.append(f"- Density raster: `{density_path}`")
    md.append(f"- Candidates: `candidates.csv`, `candidates.geojson`, `candidates.kml`")
    md.append(f"- Clusters: `{clusters_csv.name}`")
    md.append("- Run metadata: `run_params.json`")
    md.append(f"- Plots: `plots/`")
    md.append(f"- HTML: `report.html` + `html/img/`")
    md.append("")
    md.append("## Parameters used (key)")
    md.append(f"- pos_relief_threshold: **{pos_thresh:.4f} m** (spec: `{params.pos_relief_threshold_spec}`)")
    md.append(f"- min_region_pixels: **{params.min_region_pixels}**")
    md.append(f"- max_slope_deg: **{params.max_slope_deg:.1f}**")
    md.append(
        f"- consensus: enabled={params.consensus_enabled}, percentiles={[float(p) for p in params.consensus_percentiles]}, "
        f"min_support={params.consensus_min_support}, radius={params.consensus_match_radius_m:.1f}m"
    )
    md.append(f"- density_sigma_pix: **{params.density_sigma_pix}**")
    md.append(f"- min_density: **{min_density:.4f}** (spec: `{params.min_density_spec}`)")
    md.append(
        f"- post-filters: min_peak={params.min_peak_relief_m:.2f}m, min_area={params.min_area_m2:.1f}m², "
        f"max_area={params.max_area_m2:.1f}m², "
        f"min_extent={params.min_extent:.2f}, max_aspect={params.max_aspect:.2f}, edge_buffer={params.edge_buffer_m:.1f}m, "
        f"min_spacing={params.min_candidate_spacing_m:.1f}m, min_prominence={params.min_prominence_m:.2f}m, "
        f"min_compactness={params.min_compactness:.2f}, min_solidity={params.min_solidity:.2f}"
    )
    md.append(
        f"- score exponents: dens^{params.score_density_exp:.2f}, peak^{params.score_peak_exp:.2f}, "
        f"extent^{params.score_extent_exp:.2f}, consensus_support^{params.score_consensus_exp:.2f}, "
        f"prominence^{params.score_prominence_exp:.2f}, compactness^{params.score_compactness_exp:.2f}, "
        f"solidity^{params.score_solidity_exp:.2f}, area^{params.score_area_exp:.2f}"
    )
    md.append(f"- cluster_eps_mode: **{params.cluster_eps_mode}** (base={params.cluster_eps_m:.1f} m), min_samples: **{params.cluster_min_samples}**")
    md.append(f"- KML labeled top-N: **{params.kml_label_top_n}**")
    md.append("")
    md.append("## Summary")
    md.append(f"- Candidates detected: **{len(candidates)}**")
    md.append(f"- Clusters (DBSCAN): **{n_clusters}** (noise: {n_noise})")
    md.append("")
    md.append("## Top candidates")
    md.append("")
    md.append("| rank | cand_id | score | dens | peak(m) | support | prominence(m) | area(m²) | extent | aspect | compactness | solidity | cluster | lon | lat |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, c in enumerate(top, start=1):
        md.append(
            f"| {i} | {c.cand_id} | {c.score:.4f} | {c.density:.3f} | {c.peak_relief_m:.2f} | {c.consensus_support} | "
            f"{c.prominence_m:.2f} | {c.area_m2:.0f} | "
            f"{c.extent:.2f} | {c.aspect:.2f} | {c.compactness:.2f} | {c.solidity:.2f} | {c.cluster_id} | {c.lon:.6f} | {c.lat:.6f} |"
        )
    md.append("")
    md.append("## Notes")
    md.append("- Extent = **area / bbox_area** (0..1). Higher generally means more coherent/filled region.")
    md.append("- Aspect = max(width/height, height/width). Very large aspect often means linear/noisy ridges.")
    md.append("- Local prominence = **region mean relief - surrounding ring mean relief**. Low values often indicate background trends.")
    md.append("- Edge buffer drops regions near tile boundaries to reduce edge artifacts.")
    md.append("- Spacing de-dup keeps highest-score candidate within each local spacing radius.")
    md.append("- Consensus support counts how many threshold runs contain a nearby matching region.")
    md.append("- Compactness = **4πA/P²** (0..1). Lower values are more line-like and likely false positives.")
    md.append("- Solidity = **area / convex_hull_area** (0..1). Lower values are fragmented/irregular shapes.")
    md.append("- Slope filter uses **region slope q75** (not centroid slope).")
    md.append("- Candidate density uses **region mean density** over each connected region.")
    md.append("- Clustering/distances are done in **meters** (auto-UTM if source CRS is geographic).")
    md.append("- KML ‘All Candidates’ dots have label scale=0 to prevent Google Earth text overload.")
    md.append("")

    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def write_report_pdf(md_path: Path, pdf_path: Path) -> None:
    if canvas is None:
        LOG.warning("reportlab not installed; skipping PDF report.")
        return

    text = md_path.read_text(encoding="utf-8").splitlines()
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    x = 50
    y = height - 50
    c.setFont("Helvetica", 10)

    for line in text:
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50
        c.drawString(x, y, line[:110])
        y -= 12
    c.save()


def update_manifest(runs_dir: Path, run_name: str, out_dir: Path, input_path: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = runs_dir / "manifest.csv"
    new = not manifest.exists()

    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "input": str(input_path),
        "out_dir": str(out_dir),
    }

    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    LOG.info("Updated manifest: %s", manifest)


def write_run_params_json(
    out_dir: Path,
    run_name: str,
    input_path: Path,
    params: Params,
    pos_thresh: float,
    min_density: float,
    src_crs: CRS,
    clustering_crs: Optional[CRS],
    pdal_ver: str,
    dropped_edge: int,
    dropped_consensus: int,
    dropped_density: int,
    dropped_post: int,
    dropped_spacing: int,
    candidate_count: int,
    stage_metrics: Optional[Dict[str, float]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "input_path": str(input_path),
        "pdal_version": pdal_ver,
        "source_crs": src_crs.to_string(),
        "clustering_crs": None if clustering_crs is None else clustering_crs.to_string(),
        "resolved_thresholds": {
            "pos_relief_m": float(pos_thresh),
            "min_density": float(min_density),
        },
        "candidate_accounting": {
            "dropped_edge_buffer": int(dropped_edge),
            "dropped_consensus_support": int(dropped_consensus),
            "dropped_density": int(dropped_density),
            "dropped_post_filters": int(dropped_post),
            "dropped_spacing_dedup": int(dropped_spacing),
            "kept_candidates": int(candidate_count),
        },
        "stage_metrics_sec": stage_metrics or {},
        "params": asdict(params),
    }
    out_path = out_dir / "run_params.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# -----------------------------
# HTML cutouts + report
# -----------------------------
def _clamp_window(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0c = max(0, min(w, x0))
    y0c = max(0, min(h, y0))
    x1c = max(0, min(w, x1))
    y1c = max(0, min(h, y1))
    if x1c <= x0c:
        x1c = min(w, x0c + 1)
    if y1c <= y0c:
        y1c = min(h, y0c + 1)
    return x0c, y0c, x1c, y1c


def generate_candidate_panels(
    out_dir: Path,
    run_name: str,
    dtm: np.ndarray,
    lrm: np.ndarray,
    params: Params,
    candidates: List[Candidate],
    top_n: int,
    res_m: float,
) -> None:
    import matplotlib.pyplot as plt

    img_dir = out_dir / "html" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    hs = hillshade(dtm, res_m=res_m)

    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    take = sorted_c[: max(0, int(top_n))]

    half_pix = int(round((params.cutout_size_m / res_m) / 2.0))
    LOG.info("Generating cutouts: size=%.1fm (~%d px), top_n=%d", params.cutout_size_m, half_pix * 2, len(take))

    H, W = lrm.shape[:2]

    for c in take:
        cx = int(round(c.px_x))
        cy = int(round(c.px_y))

        x0, y0, x1, y1 = _clamp_window(cx - half_pix, cy - half_pix, cx + half_pix, cy + half_pix, W, H)

        lrm_crop = lrm[y0:y1, x0:x1]
        hs_crop = hs[y0:y1, x0:x1]

        vals = lrm_crop[np.isfinite(lrm_crop)]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
        else:
            lo, hi = -1.0, 1.0
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            lo, hi = -1.0, 1.0
        lrm_show = np.clip(np.where(np.isfinite(lrm_crop), lrm_crop, 0.0), lo, hi)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(lrm_show, cmap="gray", vmin=lo, vmax=hi)
        ax1.scatter([cx - x0], [cy - y0], s=18)
        ax1.set_title("LRM")
        ax1.set_axis_off()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(hs_crop, cmap="gray", vmin=0.0, vmax=1.0)
        ax2.scatter([cx - x0], [cy - y0], s=18)
        ax2.set_title("Hillshade")
        ax2.set_axis_off()

        fig.suptitle(
            f"{run_name} — cand {c.cand_id} | score {c.score:.3f} | dens {c.density:.3f} | "
            f"peak {c.peak_relief_m:.2f}m | prom {c.prominence_m:.2f}m | extent {c.extent:.2f}",
            fontsize=10,
        )

        fname = f"cand_{c.cand_id:04d}_panel.png"
        out_path = img_dir / fname
        fig.tight_layout()
        fig.savefig(out_path, dpi=int(params.cutout_dpi), bbox_inches="tight")
        plt.close(fig)

        c.img_relpath = f"html/img/{fname}"


def write_html_report(
    out_dir: Path,
    run_name: str,
    input_path: Path,
    candidates: List[Candidate],
    params: Params,
    pos_thresh: float,
    min_density: float,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not candidates:
        center_lat, center_lon = 0.0, 0.0
    else:
        center_lat = float(np.mean([c.lat for c in candidates]))
        center_lon = float(np.mean([c.lon for c in candidates]))

    sorted_c = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = sorted_c[: params.report_top_n]
    n_clusters = len({c.cluster_id for c in candidates if c.cluster_id != -1})
    n_noise = sum(1 for c in candidates if c.cluster_id == -1)
    top_score = float(sorted_c[0].score) if sorted_c else 0.0
    median_score = float(np.median([c.score for c in sorted_c])) if sorted_c else 0.0
    mean_support = float(np.mean([c.consensus_support for c in sorted_c])) if sorted_c else 0.0

    points = []
    for c in sorted_c:
        points.append(
            {
                "cand_id": c.cand_id,
                "score": c.score,
                "density": c.density,
                "peak": c.peak_relief_m,
                "support": c.consensus_support,
                "prominence": c.prominence_m,
                "area": c.area_m2,
                "extent": c.extent,
                "aspect": c.aspect,
                "compactness": c.compactness,
                "solidity": c.solidity,
                "cluster": c.cluster_id,
                "lat": c.lat,
                "lon": c.lon,
                "img": c.img_relpath or "",
            }
        )

    html_path = out_dir / "report.html"
    s_points = json.dumps(points)

    doc = f"""<!doctype html>
<html><head><meta charset='utf-8'/>
<title>MayaScan Report — {html.escape(run_name)}</title>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>

<style>
:root {{
  --ink: #111827;
  --muted: #4b5563;
  --border: #e5e7eb;
  --bg: #f7f8fa;
  --card: #ffffff;
  --accent: #0b63ce;
}}
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; color: var(--ink); background: var(--bg); }}
.wrap {{ max-width: 1240px; margin: 0 auto; padding: 18px; }}
h1 {{ margin: 0 0 8px 0; }}
h2 {{ margin: 16px 0 8px 0; }}
h3 {{ margin: 0 0 8px 0; }}
.small {{ color: var(--muted); }}
#map {{ height: 520px; border: 1px solid var(--border); border-radius: 12px; margin: 14px 0; background: #fff; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.card {{ border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: var(--card); }}
.hr {{ border-top: 1px solid var(--border); margin: 18px 0; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 10px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px; border-bottom: 1px solid var(--border); font-size: 14px; }}
th {{ text-align: left; background: #fafafa; position: sticky; top: 0; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2f7; font-size: 12px; }}
.topnote {{ margin-top: 8px; }}
.kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
.kpi {{ border: 1px solid var(--border); border-radius: 10px; background: #fff; padding: 10px; }}
.kpi .k {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
.kpi .v {{ display: block; font-size: 22px; font-weight: 700; }}
.candidate-card {{ border: 1px solid var(--border); border-radius: 12px; padding: 12px; background: #fff; margin: 0 0 14px 0; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 10px 0; }}
.metric-chip {{ border: 1px solid var(--border); border-radius: 10px; padding: 8px; background: #fafafa; }}
.metric-chip .mk {{ display: block; font-size: 11px; color: var(--muted); }}
.metric-chip .mv {{ display: block; font-size: 15px; font-weight: 600; }}
details.card summary {{ cursor: pointer; font-weight: 600; }}

@media (max-width: 1000px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  #map {{ height: 460px; }}
}}
</style>

</head><body><div class='wrap'>
<h1>MayaScan Report — {html.escape(run_name)}</h1>
<div class='small'>Timestamp: <b>{ts}</b> &nbsp;|&nbsp; Input: <code>{html.escape(str(input_path))}</code></div>
<div class='topnote small'>
pos_relief_threshold: <b>{pos_thresh:.4f} m</b> ({html.escape(params.pos_relief_threshold_spec)}) &nbsp;|&nbsp;
min_density: <b>{min_density:.4f}</b> ({html.escape(params.min_density_spec)}) &nbsp;|&nbsp;
KML labels: top <b>{params.kml_label_top_n}</b>
</div>

<div class='kpis'>
  <div class='kpi'><span class='k'>Candidates</span><span class='v'>{len(candidates)}</span></div>
  <div class='kpi'><span class='k'>Clusters</span><span class='v'>{n_clusters}</span></div>
  <div class='kpi'><span class='k'>Noise points</span><span class='v'>{n_noise}</span></div>
  <div class='kpi'><span class='k'>Top score</span><span class='v'>{top_score:.3f}</span></div>
  <div class='kpi'><span class='k'>Median score</span><span class='v'>{median_score:.3f}</span></div>
  <div class='kpi'><span class='k'>Mean support</span><span class='v'>{mean_support:.2f}</span></div>
</div>

<div id='map'></div>

<div class='grid'>
  <details class='card' open>
    <summary>How to triage</summary>
    <ol class='small'>
      <li>Scan the map for clusters and repeated platform-like forms.</li>
      <li>Click markers to review score, shape metrics, and cutout panel.</li>
      <li>Start review from top score, then confirm with LRM and hillshade texture.</li>
      <li>Use GIS exports + report assets for deeper interpretation workflow.</li>
    </ol>
  </details>
  <details class='card'>
    <summary>Files in this run</summary>
    <ul class='small'>
      <li><code>candidates.csv</code>, <code>candidates.geojson</code>, <code>candidates.kml</code></li>
      <li><code>dtm.tif</code>, <code>lrm.tif</code>, <code>mound_density.tif</code></li>
      <li><code>run_params.json</code> (resolved settings + thresholds)</li>
      <li><code>plots/</code> (density, overlay, histograms)</li>
      <li><code>html/img/</code> (candidate cutouts)</li>
    </ul>
  </details>
</div>

<div class='hr'></div>
<h2>Top candidates</h2>
<div class='small'>Click coordinates to open in Google Maps. Images show LRM + hillshade panels when available.</div>
<div class='hr'></div>
"""

    for rank, c in enumerate(top, start=1):
        gmaps = f"https://www.google.com/maps?q={c.lat:.8f},{c.lon:.8f}"
        img_tag = ""
        if c.img_relpath:
            img_tag = f"<img src='{html.escape(c.img_relpath)}' alt='candidate {c.cand_id} cutout'/>"
        doc += f"""
<div class='candidate-card'>
  <h3>Candidate {c.cand_id} <span class='badge'>rank {rank}</span> — score {c.score:.3f}</h3>
  <div class='metric-grid'>
    <div class='metric-chip'><span class='mk'>density</span><span class='mv'>{c.density:.3f}</span></div>
    <div class='metric-chip'><span class='mk'>peak (m)</span><span class='mv'>{c.peak_relief_m:.2f}</span></div>
    <div class='metric-chip'><span class='mk'>support</span><span class='mv'>{c.consensus_support}</span></div>
    <div class='metric-chip'><span class='mk'>prominence (m)</span><span class='mv'>{c.prominence_m:.2f}</span></div>
    <div class='metric-chip'><span class='mk'>area (m²)</span><span class='mv'>{c.area_m2:.0f}</span></div>
    <div class='metric-chip'><span class='mk'>extent</span><span class='mv'>{c.extent:.2f}</span></div>
    <div class='metric-chip'><span class='mk'>aspect</span><span class='mv'>{c.aspect:.2f}</span></div>
    <div class='metric-chip'><span class='mk'>compactness</span><span class='mv'>{c.compactness:.2f}</span></div>
    <div class='metric-chip'><span class='mk'>solidity</span><span class='mv'>{c.solidity:.2f}</span></div>
    <div class='metric-chip'><span class='mk'>cluster</span><span class='mv'>{c.cluster_id}</span></div>
  </div>
  <p class='small'><a href='{gmaps}' target='_blank'>{c.lat:.6f}, {c.lon:.6f}</a></p>
  {img_tag}
</div>
"""

    doc += """
<h2>All candidates</h2>
<details class='card' open>
<summary>Full candidate table (sorted by score)</summary>
<div class='small' style='margin:8px 0 10px 0;'>Map includes all points. Use this table for exhaustive review and exports.</div>
<div style='max-height:520px; overflow:auto;'>
<table>
<thead>
<tr>
<th>rank</th><th>cand_id</th><th>score</th><th>dens</th><th>peak(m)</th><th>support</th><th>prom(m)</th><th>area(m²)</th><th>extent</th><th>aspect</th><th>compact</th><th>solidity</th><th>cluster</th><th>lat</th><th>lon</th>
</tr>
</thead>
<tbody>
"""
    for rank, c in enumerate(sorted_c, start=1):
        gmaps = f"https://www.google.com/maps?q={c.lat:.8f},{c.lon:.8f}"
        doc += (
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{c.cand_id}</td>"
            f"<td>{c.score:.3f}</td>"
            f"<td>{c.density:.3f}</td>"
            f"<td>{c.peak_relief_m:.2f}</td>"
            f"<td>{c.consensus_support}</td>"
            f"<td>{c.prominence_m:.2f}</td>"
            f"<td>{c.area_m2:.0f}</td>"
            f"<td>{c.extent:.2f}</td>"
            f"<td>{c.aspect:.2f}</td>"
            f"<td>{c.compactness:.2f}</td>"
            f"<td>{c.solidity:.2f}</td>"
            f"<td>{c.cluster_id}</td>"
            f"<td><a href='{gmaps}' target='_blank'>{c.lat:.6f}</a></td>"
            f"<td><a href='{gmaps}' target='_blank'>{c.lon:.6f}</a></td>"
            "</tr>\n"
        )

    doc += f"""
</tbody></table></div></details>

<script>
const points = {s_points};

const map = L.map('map').setView([{center_lat:.8f}, {center_lon:.8f}], 14);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

function radiusFromScore(score) {{
  const r = 4 + Math.min(18, Math.sqrt(Math.max(0, score)) * 3.0);
  return r;
}}

function colorFromCluster(cid) {{
  if (cid === -1) return '#555';
  const palette = ['#b91c1c','#1d4ed8','#047857','#7c3aed','#c2410c','#0f766e','#a21caf','#4338ca'];
  return palette[(cid-1) % palette.length];
}}

const bounds = [];
points.forEach(p => {{
  const r = radiusFromScore(p.score);
  const col = colorFromCluster(p.cluster);
  const gmaps = `https://www.google.com/maps?q=${{p.lat}},${{p.lon}}`;

  let imgHtml = '';
  if (p.img && p.img.length > 0) {{
    imgHtml = `<div style="margin-top:8px"><img src="${{p.img}}" style="max-width:260px; border:1px solid #ddd; border-radius:8px"/></div>`;
  }}

  const popup = `
    <div style="font-size:14px">
      <b>Candidate ${{p.cand_id}}</b><br/>
      score <b>${{p.score.toFixed(3)}}</b> | dens ${{p.density.toFixed(3)}}<br/>
      peak ${{p.peak.toFixed(2)}}m | support ${{p.support}} | prom ${{p.prominence.toFixed(2)}}m | area ${{Math.round(p.area)}} m²<br/>
      extent ${{p.extent.toFixed(2)}} | aspect ${{p.aspect.toFixed(2)}}<br/>
      compactness ${{p.compactness.toFixed(2)}} | solidity ${{p.solidity.toFixed(2)}}<br/>
      cluster ${{p.cluster}}<br/>
      <a href="${{gmaps}}" target="_blank">Open in Google Maps</a>
      ${{imgHtml}}
    </div>
  `;

  const marker = L.circleMarker([p.lat, p.lon], {{
    radius: r,
    color: col,
    weight: 2,
    fillColor: col,
    fillOpacity: 0.55
  }}).addTo(map);
  marker.bindPopup(popup);
  bounds.push([p.lat, p.lon]);
}});

if (bounds.length > 0) {{
  map.fitBounds(bounds, {{padding:[20,20]}});
}}
</script>

</div></body></html>
"""
    html_path.write_text(doc, encoding="utf-8")
    return html_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="MayaScan: LiDAR archaeology discovery pipeline")

    ap.add_argument("-i", "--input", required=True, help="Input LAZ/LAS file")
    ap.add_argument("--name", required=True, help="Run name (folder under runs/)")
    ap.add_argument("--runs-dir", default="runs", help="Runs base directory")
    ap.add_argument("--overwrite", action="store_true", help="Allow deleting an existing run folder")

    ap.add_argument("--try-smrf", action="store_true", help="Try PDAL SMRF ground classification before DTM")

    # knobs
    ap.add_argument("--pos-thresh", type=_arg_pos_thresh_spec, default=None, help="Override pos relief threshold (e.g. 0.20 or auto:p96)")
    ap.add_argument("--min-density", type=_arg_min_density_spec, default=None, help="Override min density threshold (e.g. 0.10 or auto:p60)")
    ap.add_argument("--density-sigma", type=_arg_positive_float, default=None, help="Override density sigma (pixels)")
    ap.add_argument("--max-slope-deg", type=_arg_nonnegative_float, default=None, help="Max allowed region slope q75 in degrees (default 20)")
    ap.add_argument("--no-consensus", action="store_true", help="Disable multi-threshold consensus support filtering")
    ap.add_argument(
        "--consensus-percentiles",
        type=_arg_percentiles_csv,
        default=None,
        help="Comma-separated auto percentiles for consensus support (default 95,96,97)",
    )
    ap.add_argument(
        "--consensus-min-support",
        type=_arg_positive_int,
        default=None,
        help="Minimum threshold-support count required to keep a region (default 2)",
    )
    ap.add_argument(
        "--consensus-radius-m",
        type=_arg_nonnegative_float,
        default=None,
        help="Match radius (m) when counting support across thresholds (default 12)",
    )

    # post-filters
    ap.add_argument("--min-peak", type=_arg_nonnegative_float, default=None, help="Min peak relief (m) post-filter (e.g. 0.5)")
    ap.add_argument("--min-area-m2", type=_arg_nonnegative_float, default=None, help="Min area (m^2) post-filter (e.g. 30)")
    ap.add_argument("--max-area-m2", type=_arg_nonnegative_float, default=None, help="Max area (m^2) post-filter; <=0 disables")
    ap.add_argument("--min-extent", type=_arg_unit_interval, default=None, help="Min extent (0..1) post-filter (e.g. 0.35)")
    ap.add_argument("--max-aspect", type=_arg_ge_one_float, default=None, help="Max aspect ratio post-filter (e.g. 4.0)")
    ap.add_argument("--edge-buffer-m", type=_arg_nonnegative_float, default=None, help="Drop regions near tile edge within this distance (m, default 10)")
    ap.add_argument("--min-spacing-m", type=_arg_nonnegative_float, default=None, help="Score-ordered minimum spacing between candidates (m, default 15)")
    ap.add_argument("--prominence-ring-pix", type=_arg_positive_int, default=None, help="Ring width (pixels) for local prominence estimate (default 6)")
    ap.add_argument("--min-prominence", type=_arg_nonnegative_float, default=None, help="Min local prominence (m) post-filter (default 0.10)")
    ap.add_argument("--min-compactness", type=_arg_unit_interval, default=None, help="Min compactness 4*pi*A/P^2 (0..1), lower removes line-like shapes")
    ap.add_argument("--min-solidity", type=_arg_unit_interval, default=None, help="Min solidity area/convex_hull_area (0..1), lower removes fragmented/linear shapes")

    # scoring knobs
    ap.add_argument("--score-extent-exp", type=_arg_nonnegative_float, default=None, help="Exponent for extent in score (default 0.35)")
    ap.add_argument("--score-consensus-exp", type=_arg_nonnegative_float, default=None, help="Exponent for consensus support in score (default 0.40)")
    ap.add_argument("--score-prominence-exp", type=_arg_nonnegative_float, default=None, help="Exponent for prominence in score (default 0.75)")
    ap.add_argument("--score-compactness-exp", type=_arg_nonnegative_float, default=None, help="Exponent for compactness in score (default 0.20)")
    ap.add_argument("--score-solidity-exp", type=_arg_nonnegative_float, default=None, help="Exponent for solidity in score (default 0.20)")
    ap.add_argument("--score-area-exp", type=_arg_nonnegative_float, default=None, help="Exponent for area_m2 in score (default 0.50)")

    # clustering knobs
    ap.add_argument("--cluster-eps", type=_arg_cluster_eps_spec, default=None, help="DBSCAN eps in meters or 'auto' (default auto)")
    ap.add_argument("--min-samples", type=_arg_positive_int, default=None, help="DBSCAN min_samples (default 3)")

    ap.add_argument("--label-top-n", type=_arg_nonnegative_int, default=None, help="Override KML labeled top-N")
    ap.add_argument("--report-top-n", type=_arg_nonnegative_int, default=None, help="Override report top-N table size")

    # HTML / cutouts
    ap.add_argument("--no-html", action="store_true", help="Disable HTML report + cutout images")
    ap.add_argument("--cutout-size-m", type=_arg_positive_float, default=None, help="Cutout panel window size in meters (default 140)")
    ap.add_argument("--cutout-top-n", type=_arg_nonnegative_int, default=None, help="How many top candidates get cutouts (default report_top_n)")

    args = ap.parse_args()

    try:
        run_name = sanitize_run_name(args.name)
    except ValueError as exc:
        ap.error(str(exc))
    if run_name != args.name:
        print(f"Run name sanitized: '{args.name}' -> '{run_name}'", file=sys.stderr)

    params = Params()

    if args.pos_thresh is not None:
        params.pos_relief_threshold_spec = args.pos_thresh
    if args.min_density is not None:
        params.min_density_spec = args.min_density
    if args.density_sigma is not None:
        params.density_sigma_pix = args.density_sigma
    if args.max_slope_deg is not None:
        params.max_slope_deg = args.max_slope_deg
    if args.no_consensus:
        params.consensus_enabled = False
    if args.consensus_percentiles is not None:
        vals = tuple(float(x) for x in str(args.consensus_percentiles).split(",") if x)
        if vals:
            params.consensus_percentiles = vals
    if args.consensus_min_support is not None:
        params.consensus_min_support = int(args.consensus_min_support)
    if args.consensus_radius_m is not None:
        params.consensus_match_radius_m = float(args.consensus_radius_m)

    if args.min_peak is not None:
        params.min_peak_relief_m = args.min_peak
    if args.min_area_m2 is not None:
        params.min_area_m2 = args.min_area_m2
    if args.max_area_m2 is not None:
        params.max_area_m2 = args.max_area_m2
    if args.min_extent is not None:
        params.min_extent = args.min_extent
    if args.max_aspect is not None:
        params.max_aspect = args.max_aspect
    if args.edge_buffer_m is not None:
        params.edge_buffer_m = args.edge_buffer_m
    if args.min_spacing_m is not None:
        params.min_candidate_spacing_m = args.min_spacing_m
    if args.prominence_ring_pix is not None:
        params.prominence_ring_pixels = args.prominence_ring_pix
    if args.min_prominence is not None:
        params.min_prominence_m = args.min_prominence
    if args.min_compactness is not None:
        params.min_compactness = args.min_compactness
    if args.min_solidity is not None:
        params.min_solidity = args.min_solidity

    if args.score_extent_exp is not None:
        params.score_extent_exp = args.score_extent_exp
    if args.score_consensus_exp is not None:
        params.score_consensus_exp = args.score_consensus_exp
    if args.score_prominence_exp is not None:
        params.score_prominence_exp = args.score_prominence_exp
    if args.score_compactness_exp is not None:
        params.score_compactness_exp = args.score_compactness_exp
    if args.score_solidity_exp is not None:
        params.score_solidity_exp = args.score_solidity_exp
    if args.score_area_exp is not None:
        params.score_area_exp = args.score_area_exp

    if args.cluster_eps is not None:
        if args.cluster_eps == "auto":
            params.cluster_eps_mode = "auto"
        else:
            params.cluster_eps_mode = "fixed"
            params.cluster_eps_m = float(args.cluster_eps)

    if args.min_samples is not None:
        params.cluster_min_samples = args.min_samples

    if args.label_top_n is not None:
        params.kml_label_top_n = args.label_top_n
    if args.report_top_n is not None:
        params.report_top_n = args.report_top_n

    if args.no_html:
        params.html_report = False
    if args.cutout_size_m is not None:
        params.cutout_size_m = args.cutout_size_m

    input_path = Path(args.input).expanduser().resolve()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = (runs_dir / run_name).resolve()
    ensure_path_within(runs_dir, out_dir)

    if out_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Run dir already exists: {out_dir}\n"
                f"Use --overwrite if you want to delete/recreate it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    pdal_ver = pdal_version()
    LOG.info("PDAL detected: %s", pdal_ver)
    run_t0 = time.perf_counter()
    stage_metrics: Dict[str, float] = {}

    dtm_path = out_dir / "dtm.tif"
    lrm_path = out_dir / "lrm.tif"
    density_path = out_dir / "mound_density.tif"
    geojson_path = out_dir / "candidates.geojson"
    kml_path = out_dir / "candidates.kml"
    csv_path = out_dir / "candidates.csv"
    clusters_csv = out_dir / "clusters.csv"
    report_pdf = out_dir / "report.pdf"
    html_report_path = out_dir / "report.html"

    tmp_dir = out_dir / "_tmp"

    LOG.info("Step 0: Building DTM from LAZ/LAS")
    step_t0 = time.perf_counter()
    build_dtm_from_laz(
        laz_path=input_path,
        out_dtm_tif=dtm_path,
        tmp_dir=tmp_dir,
        resolution_m=params.dtm_resolution_m,
        try_smrf=bool(args.try_smrf),
    )
    stage_metrics["step0_dtm_sec"] = float(time.perf_counter() - step_t0)
    LOG.info("DTM written: %s", dtm_path)

    LOG.info("Step 1: Building multi-scale LRM")
    step_t0 = time.perf_counter()
    dtm, dtm_prof = load_raster(dtm_path)
    dtm_transform = dtm_prof["transform"]
    res_m = _res_m_from_profile(dtm_prof)

    slope_deg = compute_slope_degrees(dtm, res_m=float(res_m))
    lrm = build_multiscale_lrm(dtm, params)
    write_float_geotiff(lrm_path, lrm, dtm_prof)
    stage_metrics["step1_lrm_sec"] = float(time.perf_counter() - step_t0)
    LOG.info("LRM written: %s", lrm_path)

    LOG.info("Step 2: Detecting candidate structures")
    step_t0 = time.perf_counter()
    LOG.info(
        "Consensus config: enabled=%s | percentiles=%s | min_support=%d | radius=%.1fm",
        str(params.consensus_enabled),
        ",".join(f"{p:g}" for p in params.consensus_percentiles),
        int(params.consensus_min_support),
        float(params.consensus_match_radius_m),
    )
    regions, density_norm, pos_thresh, min_density, detect_diag = detect_candidates(
        lrm=lrm,
        dtm_slope_deg=slope_deg,
        profile=dtm_prof,
        params=params,
        out_density_tif=density_path,
    )
    stage_metrics["step2_detect_sec"] = float(time.perf_counter() - step_t0)

    crs_any = dtm_prof.get("crs")
    if crs_any is None:
        raise RuntimeError("DTM has no CRS; cannot export lon/lat")
    src_crs = CRS.from_user_input(crs_any)
    transformer_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    # Candidate build + filters
    candidates: List[Candidate] = []
    dropped_density = 0
    dropped_post = 0
    dropped_edge = 0
    dropped_spacing = 0
    dropped_consensus = int(detect_diag.get("consensus_dropped", 0))
    edge_buffer_pix = int(math.ceil(max(0.0, float(params.edge_buffer_m)) / max(1e-9, float(res_m))))
    H, W = density_norm.shape

    cand_id = 1
    for r in regions:
        if edge_buffer_pix > 0:
            x0 = int(r.get("x0", 0))
            y0 = int(r.get("y0", 0))
            x1 = int(r.get("x1", W - 1))
            y1 = int(r.get("y1", H - 1))
            if x0 <= edge_buffer_pix or y0 <= edge_buffer_pix or x1 >= (W - 1 - edge_buffer_pix) or y1 >= (H - 1 - edge_buffer_pix):
                dropped_edge += 1
                continue

        dens = float(r.get("density_mean", np.nan))
        if not np.isfinite(dens):
            dropped_density += 1
            continue
        if dens < float(min_density):
            dropped_density += 1
            continue

        peak = float(r["peak"])
        area_m2 = float(r["area_m2"])
        extent = float(r["extent"])
        aspect = float(r["aspect"])
        consensus_support = int(max(1, int(r.get("consensus_support", 1))))
        prominence = float(r.get("prominence_m", 0.0))
        compactness = float(r.get("compactness", 0.0))
        solidity = float(r.get("solidity", 0.0))

        # post-filters for “project goal”
        if (
            peak < params.min_peak_relief_m
            or area_m2 < params.min_area_m2
            or (params.max_area_m2 > 0.0 and area_m2 > params.max_area_m2)
            or extent < params.min_extent
            or aspect > params.max_aspect
            or prominence < params.min_prominence_m
            or compactness < params.min_compactness
            or solidity < params.min_solidity
        ):
            dropped_post += 1
            continue

        score = (
            (dens ** params.score_density_exp)
            * (max(1e-9, peak) ** params.score_peak_exp)
            * ((max(1e-6, extent)) ** params.score_extent_exp)
            * ((max(1.0, float(consensus_support))) ** params.score_consensus_exp)
            * ((max(1e-6, prominence)) ** params.score_prominence_exp)
            * ((max(1e-6, compactness)) ** params.score_compactness_exp)
            * ((max(1e-6, solidity)) ** params.score_solidity_exp)
            * (max(1e-9, area_m2) ** params.score_area_exp)
        )

        x_map, y_map = pix2map_xy(dtm_transform, r["cy"], r["cx"])
        lon, lat = transformer_ll.transform(x_map, y_map)

        candidates.append(
            Candidate(
                cand_id=cand_id,
                px_x=float(r["cx"]),
                px_y=float(r["cy"]),
                peak_relief_m=peak,
                mean_relief_m=float(r["mean"]),
                area_m2=area_m2,
                density=dens,
                extent=extent,
                aspect=aspect,
                consensus_support=consensus_support,
                prominence_m=prominence,
                compactness=compactness,
                solidity=solidity,
                width_m=float(r["width_m"]),
                height_m=float(r["height_m"]),
                score=float(score),
                lon=float(lon),
                lat=float(lat),
            )
        )
        cand_id += 1

    if candidates and params.min_candidate_spacing_m > 0:
        candidates, dropped_spacing = dedupe_candidates_by_spacing(
            candidates=candidates,
            src_crs=src_crs,
            dtm_transform=dtm_transform,
            min_spacing_m=float(params.min_candidate_spacing_m),
        )
        for i, c in enumerate(candidates, start=1):
            c.cand_id = i

    LOG.info("Dropped by edge buffer (%.1f m): %d", params.edge_buffer_m, dropped_edge)
    LOG.info("Dropped by consensus support: %d", dropped_consensus)
    LOG.info("Dropped by density (region mean < min_density): %d", dropped_density)
    LOG.info("Dropped by post-filters: %d", dropped_post)
    LOG.info("Dropped by spacing de-dup (%.1f m): %d", params.min_candidate_spacing_m, dropped_spacing)
    LOG.info("Kept candidates after density + post-filters: %d", len(candidates))

    LOG.info("Step 3: Clustering + settlement pattern analysis (meters)")
    step_t0 = time.perf_counter()
    used_m_crs: Optional[CRS] = None
    if candidates:
        xs = []
        ys = []
        for c in candidates:
            x_map, y_map = pix2map_xy(dtm_transform, c.px_y, c.px_x)
            xs.append(float(x_map))
            ys.append(float(y_map))
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        xs_m, ys_m, used_m_crs = project_points_to_meters(src_crs, xs, ys)
        LOG.info("Clustering CRS: %s", used_m_crs.to_string())

        labels = cluster_candidates_meters(xs_m, ys_m, params)
        for c, lab in zip(candidates, labels):
            c.cluster_id = int(lab)

        assign_cluster_core_distances(candidates, xs_m, ys_m)

        n_clusters = len({c.cluster_id for c in candidates if c.cluster_id != -1})
        LOG.info("Clusters found: %d (noise=%d)", n_clusters, sum(1 for c in candidates if c.cluster_id == -1))
    stage_metrics["step3_cluster_sec"] = float(time.perf_counter() - step_t0)

    LOG.info("Step 4: Exporting GIS products")
    step_t0 = time.perf_counter()
    write_geojson(candidates, geojson_path)
    LOG.info("Wrote GeoJSON: %s", geojson_path)

    write_kml(candidates, kml_path, label_top_n=params.kml_label_top_n)
    LOG.info("Wrote KML: %s", kml_path)

    write_csv(candidates, csv_path)
    LOG.info("Wrote CSV: %s", csv_path)

    write_clusters_csv(candidates, clusters_csv)
    LOG.info("Wrote clusters CSV: %s", clusters_csv)
    stage_metrics["step4_export_sec"] = float(time.perf_counter() - step_t0)

    LOG.info("Step 5: Writing plots")
    step_t0 = time.perf_counter()
    make_plots(out_dir, lrm, density_norm, candidates)
    stage_metrics["step5_plots_sec"] = float(time.perf_counter() - step_t0)

    LOG.info("Step 6: Writing reports")
    step_t0 = time.perf_counter()
    md_path = write_report_md(
        out_dir=out_dir,
        run_name=run_name,
        input_path=input_path,
        dtm_path=dtm_path,
        lrm_path=lrm_path,
        density_path=density_path,
        candidates=candidates,
        clusters_csv=clusters_csv,
        params=params,
        pos_thresh=pos_thresh,
        min_density=min_density,
    )
    LOG.info("Wrote report.md: %s", md_path)

    write_report_pdf(md_path, report_pdf)
    if report_pdf.exists():
        LOG.info("Wrote report.pdf: %s", report_pdf)
    stage_metrics["step6_reports_sec"] = float(time.perf_counter() - step_t0)

    if params.html_report and candidates:
        cutout_top_n = params.report_top_n if args.cutout_top_n is None else int(args.cutout_top_n)
        LOG.info("Step 7: Generating HTML report + cutouts")
        step_t0 = time.perf_counter()
        generate_candidate_panels(
            out_dir=out_dir,
            run_name=run_name,
            dtm=dtm,
            lrm=lrm,
            params=params,
            candidates=candidates,
            top_n=cutout_top_n,
            res_m=float(res_m),
        )
        html_out = write_html_report(
            out_dir=out_dir,
            run_name=run_name,
            input_path=input_path,
            candidates=candidates,
            params=params,
            pos_thresh=pos_thresh,
            min_density=min_density,
        )
        LOG.info("Wrote report.html: %s", html_out)
        stage_metrics["step7_html_sec"] = float(time.perf_counter() - step_t0)

    stage_metrics["total_runtime_sec"] = float(time.perf_counter() - run_t0)
    params_json = write_run_params_json(
        out_dir=out_dir,
        run_name=run_name,
        input_path=input_path,
        params=params,
        pos_thresh=pos_thresh,
        min_density=min_density,
        src_crs=src_crs,
        clustering_crs=used_m_crs,
        pdal_ver=pdal_ver,
        dropped_edge=dropped_edge,
        dropped_consensus=dropped_consensus,
        dropped_density=dropped_density,
        dropped_post=dropped_post,
        dropped_spacing=dropped_spacing,
        candidate_count=len(candidates),
        stage_metrics=stage_metrics,
    )
    LOG.info("Wrote run_params.json: %s", params_json)

    update_manifest(runs_dir, run_name, out_dir, input_path)

    LOG.info("DONE. Output folder: %s", out_dir)
    LOG.info("Quick open: %s", kml_path)
    if params.html_report:
        LOG.info("HTML report: %s", html_report_path)
    LOG.info("Runtime summary (sec): %s", json.dumps(stage_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
