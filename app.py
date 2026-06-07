#!/usr/bin/env python3
import base64
import json
import re
import sys
import time
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = REPO_ROOT / "maya_scan.py"
LOGO_PATH = REPO_ROOT / "assets" / "mayascan_logo.svg"

PRESET_BALANCED = "Balanced (Recommended)"
PRESET_STRICT = "Strict (Low false positives)"
PRESET_EXPLORATORY = "Exploratory (High recall)"

PRESET_VALUES: dict[str, dict[str, object]] = {
    PRESET_BALANCED: {
        "cfg_pos_thresh": "auto:p96",
        "cfg_min_density": "auto:p60",
        "cfg_density_sigma": 40.0,
        "cfg_max_slope_deg": 20.0,
        "cfg_consensus_enabled": True,
        "cfg_consensus_percentiles": "95,96,97",
        "cfg_consensus_min_support": 2,
        "cfg_consensus_radius_m": 12.0,
        "cfg_min_peak": 0.50,
        "cfg_min_area_m2": 25.0,
        "cfg_max_area_m2": 1200.0,
        "cfg_min_extent": 0.38,
        "cfg_max_aspect": 3.5,
        "cfg_edge_buffer_m": 10.0,
        "cfg_min_spacing_m": 15.0,
        "cfg_min_prominence": 0.10,
        "cfg_min_compactness": 0.20,
        "cfg_min_solidity": 0.50,
        "cfg_max_peak_offset": 0.60,
        "cfg_tpi_radius_m": 50.0,
        "cfg_max_tpi": 2.0,
        "cfg_cluster_eps": "auto",
        "cfg_min_samples": 4,
        "cfg_report_top_n": 30,
        "cfg_label_top_n": 60,
    },
    PRESET_STRICT: {
        "cfg_pos_thresh": "auto:p97",
        "cfg_min_density": "auto:p65",
        "cfg_density_sigma": 45.0,
        "cfg_max_slope_deg": 18.0,
        "cfg_consensus_enabled": True,
        "cfg_consensus_percentiles": "95,96,97",
        "cfg_consensus_min_support": 2,
        "cfg_consensus_radius_m": 10.0,
        "cfg_min_peak": 0.60,
        "cfg_min_area_m2": 35.0,
        "cfg_max_area_m2": 900.0,
        "cfg_min_extent": 0.42,
        "cfg_max_aspect": 3.0,
        "cfg_edge_buffer_m": 12.0,
        "cfg_min_spacing_m": 18.0,
        "cfg_min_prominence": 0.14,
        "cfg_min_compactness": 0.22,
        "cfg_min_solidity": 0.58,
        "cfg_max_peak_offset": 0.50,
        "cfg_tpi_radius_m": 50.0,
        "cfg_max_tpi": 1.5,
        "cfg_cluster_eps": "auto",
        "cfg_min_samples": 5,
        "cfg_report_top_n": 30,
        "cfg_label_top_n": 60,
    },
    PRESET_EXPLORATORY: {
        "cfg_pos_thresh": "auto:p95",
        "cfg_min_density": "auto:p50",
        "cfg_density_sigma": 35.0,
        "cfg_max_slope_deg": 22.0,
        "cfg_consensus_enabled": True,
        "cfg_consensus_percentiles": "94,95,96",
        "cfg_consensus_min_support": 2,
        "cfg_consensus_radius_m": 14.0,
        "cfg_min_peak": 0.35,
        "cfg_min_area_m2": 15.0,
        "cfg_max_area_m2": 1800.0,
        "cfg_min_extent": 0.30,
        "cfg_max_aspect": 4.5,
        "cfg_edge_buffer_m": 8.0,
        "cfg_min_spacing_m": 10.0,
        "cfg_min_prominence": 0.05,
        "cfg_min_compactness": 0.12,
        "cfg_min_solidity": 0.40,
        "cfg_max_peak_offset": 0.70,
        "cfg_tpi_radius_m": 50.0,
        "cfg_max_tpi": 0.0,
        "cfg_cluster_eps": "auto",
        "cfg_min_samples": 3,
        "cfg_report_top_n": 30,
        "cfg_label_top_n": 60,
    },
}


def apply_preset_to_session(preset_name: str, *, update_selector: bool = True) -> None:
    values = PRESET_VALUES.get(preset_name, PRESET_VALUES[PRESET_BALANCED])
    for key, val in values.items():
        st.session_state[key] = val
    if update_selector:
        st.session_state["cfg_preset"] = preset_name


def preset_matches_session(preset_name: str) -> bool:
    values = PRESET_VALUES.get(preset_name, PRESET_VALUES[PRESET_BALANCED])
    for key, val in values.items():
        cur = st.session_state.get(key)
        if isinstance(val, float):
            try:
                if abs(float(cur) - val) > 1e-9:
                    return False
            except Exception:
                return False
        else:
            if str(cur) != str(val):
                return False
    return True


def current_ui_config_snapshot() -> dict[str, object]:
    keys = [
        "cfg_pos_thresh",
        "cfg_min_density",
        "cfg_density_sigma",
        "cfg_max_slope_deg",
        "cfg_consensus_enabled",
        "cfg_consensus_percentiles",
        "cfg_consensus_min_support",
        "cfg_consensus_radius_m",
        "cfg_min_peak",
        "cfg_min_area_m2",
        "cfg_max_area_m2",
        "cfg_min_extent",
        "cfg_max_aspect",
        "cfg_edge_buffer_m",
        "cfg_min_spacing_m",
        "cfg_min_prominence",
        "cfg_min_compactness",
        "cfg_min_solidity",
        "cfg_cluster_eps",
        "cfg_min_samples",
        "cfg_report_top_n",
        "cfg_label_top_n",
        "cfg_annotation_mode",
        "cfg_resolution_m",
        "cfg_smrf_scalar",
        "cfg_smrf_slope",
        "cfg_smrf_threshold",
        "cfg_smrf_window",
        "cfg_max_peak_offset",
        "cfg_tpi_radius_m",
        "cfg_max_tpi",
        "cfg_detect_depressions",
        "cfg_validate_against",
        "cfg_validate_match_radius_m",
    ]
    out: dict[str, object] = {}
    for key in keys:
        out[key] = st.session_state.get(key)
    return out


def preset_slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "preset"


def read_run_summary(run_dir: Path) -> dict[str, object]:
    out: dict[str, object] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "candidates": None,
        "clusters": None,
        "noise": None,
        "top_score": None,
        "mean_score": None,
        "median_score": None,
    }

    vals = parse_values_used(read_text_safely(run_dir / "process.log"))
    if "candidates_kept" in vals:
        out["candidates"] = vals.get("candidates_kept")
    if "clusters_found" in vals:
        out["clusters"] = vals.get("clusters_found")
    if "clusters_noise" in vals:
        out["noise"] = vals.get("clusters_noise")

    csv_path = run_dir / "candidates.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "score" in df.columns:
                s = pd.to_numeric(df["score"], errors="coerce").dropna()
                if not s.empty:
                    out["top_score"] = float(s.max())
                    out["mean_score"] = float(s.mean())
                    out["median_score"] = float(s.median())
            if out["candidates"] is None:
                out["candidates"] = int(len(df))
        except Exception:
            pass
    return out


def write_preset_compare_artifacts(
    runs_dir: Path,
    base_name: str,
    compare_payload: dict[str, object],
) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{base_name}_preset_compare_{stamp}"
    json_path = runs_dir / f"{stem}.json"
    md_path = runs_dir / f"{stem}.md"

    json_path.write_text(json.dumps(compare_payload, indent=2), encoding="utf-8")

    runs = compare_payload.get("runs", [])
    baseline = compare_payload.get("baseline_preset", "")
    lines: list[str] = []
    lines.append(f"# MayaScan Preset Comparison: {base_name}")
    lines.append("")
    lines.append(f"- Timestamp: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
    lines.append(f"- Baseline preset: **{baseline}**")
    lines.append("")
    lines.append("| preset | run_name | candidates | clusters | noise | top_score | mean_score | median_score | d_candidates | d_top_score |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    if isinstance(runs, list):
        for r in runs:
            if not isinstance(r, dict):
                continue
            lines.append(
                f"| {r.get('preset','')} | {r.get('run_name','')} | {r.get('candidates','')} | {r.get('clusters','')} | "
                f"{r.get('noise','')} | {r.get('top_score','')} | {r.get('mean_score','')} | {r.get('median_score','')} | "
                f"{r.get('d_candidates_vs_baseline','')} | {r.get('d_top_score_vs_baseline','')} |"
            )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def resolve_input_path(mode: str, uploaded, input_local: str) -> Path | None:
    if mode == "Upload .laz/.las":
        if uploaded is None:
            st.error("Please upload a .laz or .las file.")
            return None
        data_dir = REPO_ROOT / "data" / "lidar"
        data_dir.mkdir(parents=True, exist_ok=True)
        uploaded_safe = sanitize_filename(uploaded.name)
        upload_stamp = unique_stamp()
        input_path = data_dir / f"{upload_stamp}_{uploaded_safe}"
        input_path.write_bytes(uploaded.getvalue())
        return input_path

    input_path = Path(input_local).expanduser()
    if not input_path.is_absolute():
        input_path = (REPO_ROOT / input_path).resolve()
    if not input_path.exists():
        st.error(f"Input file not found: {input_path}")
        return None
    return input_path


# -----------------------------
# CLI command builder
# -----------------------------
def build_cmd(
    input_path: Path,
    run_name: str,
    runs_dir: Path,
    overwrite: bool,
    try_smrf: bool,
    pos_thresh: str,
    min_density: str,
    density_sigma: float,
    max_slope_deg: float,
    consensus_enabled: bool,
    consensus_percentiles: str,
    consensus_min_support: int,
    consensus_radius_m: float,
    min_peak: float,
    min_area_m2: float,
    max_area_m2: float,
    min_extent: float,
    max_aspect: float,
    edge_buffer_m: float,
    min_spacing_m: float,
    min_prominence: float,
    min_compactness: float,
    min_solidity: float,
    cluster_eps: str,
    min_samples: int,
    report_top_n: int,
    label_top_n: int,
    no_html: bool,
    resolution_m: float | None = None,
    smrf_scalar: float | None = None,
    smrf_slope: float | None = None,
    smrf_threshold: float | None = None,
    smrf_window: float | None = None,
    max_peak_offset: float = 0.0,
    tpi_radius_m: float = 50.0,
    max_tpi: float = 0.0,
    detect_depressions: bool = False,
    validate_against: str = "",
    validate_match_radius_m: float = 25.0,
):
    cmd = [sys.executable, str(SCRIPT_PATH)]
    cmd += ["-i", str(input_path)]
    cmd += ["--name", run_name]
    cmd += ["--runs-dir", str(runs_dir)]

    if overwrite:
        cmd += ["--overwrite"]
    if try_smrf:
        cmd += ["--try-smrf"]
    if no_html:
        cmd += ["--no-html"]

    if resolution_m is not None and resolution_m > 0:
        cmd += ["--resolution-m", str(resolution_m)]

    if try_smrf:
        if smrf_scalar is not None:
            cmd += ["--smrf-scalar", str(smrf_scalar)]
        if smrf_slope is not None:
            cmd += ["--smrf-slope", str(smrf_slope)]
        if smrf_threshold is not None:
            cmd += ["--smrf-threshold", str(smrf_threshold)]
        if smrf_window is not None:
            cmd += ["--smrf-window", str(smrf_window)]

    if pos_thresh.strip():
        cmd += ["--pos-thresh", pos_thresh.strip()]
    if min_density.strip():
        cmd += ["--min-density", min_density.strip()]
    if density_sigma is not None:
        cmd += ["--density-sigma", str(density_sigma)]
    if max_slope_deg is not None:
        cmd += ["--max-slope-deg", str(max_slope_deg)]
    if not consensus_enabled:
        cmd += ["--no-consensus"]
    else:
        if consensus_percentiles.strip():
            cmd += ["--consensus-percentiles", consensus_percentiles.strip()]
        if consensus_min_support is not None:
            cmd += ["--consensus-min-support", str(consensus_min_support)]
        if consensus_radius_m is not None:
            cmd += ["--consensus-radius-m", str(consensus_radius_m)]

    cmd += ["--min-peak", str(min_peak)]
    cmd += ["--min-area-m2", str(min_area_m2)]
    cmd += ["--max-area-m2", str(max_area_m2)]
    cmd += ["--min-extent", str(min_extent)]
    cmd += ["--max-aspect", str(max_aspect)]
    cmd += ["--edge-buffer-m", str(edge_buffer_m)]
    cmd += ["--min-spacing-m", str(min_spacing_m)]
    cmd += ["--min-prominence", str(min_prominence)]
    cmd += ["--min-compactness", str(min_compactness)]
    cmd += ["--min-solidity", str(min_solidity)]

    cmd += ["--max-peak-offset", str(max_peak_offset)]
    cmd += ["--tpi-radius-m", str(tpi_radius_m)]
    if max_tpi > 0.0:
        cmd += ["--max-tpi", str(max_tpi)]

    if detect_depressions:
        cmd += ["--detect-depressions"]

    if cluster_eps.strip():
        cmd += ["--cluster-eps", cluster_eps.strip()]
    cmd += ["--min-samples", str(min_samples)]

    cmd += ["--report-top-n", str(report_top_n)]
    cmd += ["--label-top-n", str(label_top_n)]

    if validate_against.strip():
        validate_path = Path(validate_against.strip()).expanduser()
        if not validate_path.is_absolute():
            validate_path = (REPO_ROOT / validate_path).resolve()
        if validate_path.exists():
            cmd += ["--validate-against", str(validate_path)]
            cmd += ["--validate-match-radius-m", str(validate_match_radius_m)]

    return cmd


# -----------------------------
# Utilities
# -----------------------------
def resolve_runs_dir(runs_dir_value: str) -> Path:
    p = Path(runs_dir_value).expanduser()
    return (REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return cleaned[:120]


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)
    cleaned = cleaned.strip("._-")
    return cleaned or "input.laz"


def unique_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def next_default_run_name() -> str:
    return f"run_{unique_stamp()}"


def next_unique_run_name(runs_dir: Path, base_name: str) -> str:
    """
    Return a filesystem-safe run name that does not already exist under runs_dir.
    """
    safe_base = sanitize_run_name(base_name)
    candidate = safe_base
    if not (runs_dir / candidate).exists():
        return candidate

    stamp = unique_stamp()
    candidate = sanitize_run_name(f"{safe_base}_{stamp}")
    if not (runs_dir / candidate).exists():
        return candidate

    for i in range(1, 1000):
        candidate = sanitize_run_name(f"{safe_base}_{stamp}_{i}")
        if not (runs_dir / candidate).exists():
            return candidate

    return sanitize_run_name(f"{safe_base}_{unique_stamp()}")


def normalize_compare_presets(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """
    Keep only valid presets, deduplicate, preserve order.
    """
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v in PRESET_VALUES and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def list_existing_runs(runs_dir_path: Path) -> list[str]:
    if not runs_dir_path.exists():
        return []
    try:
        runs = [p for p in runs_dir_path.iterdir() if p.is_dir()]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.name for p in runs]
    except Exception:
        return []


def summarize_run_option(run_dir: Path) -> str:
    name = run_dir.name
    try:
        ts = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts = "unknown time"

    candidates = "?"
    process_log = run_dir / "process.log"
    if process_log.exists():
        vals = parse_values_used(read_text_safely(process_log))
        if "candidates_kept" in vals:
            candidates = str(vals["candidates_kept"])

    input_name = "input unknown"
    report_md = run_dir / "report.md"
    if report_md.exists():
        m = re.search(r"- Input:\s*`([^`]+)`", read_text_safely(report_md))
        if m:
            input_name = Path(m.group(1)).name

    return f"{name} | {ts} | cands: {candidates} | {input_name}"


def parse_cmd_settings(cmd: list[str]) -> dict:
    out: dict[str, object] = {}
    if not cmd:
        return out
    boolean_flags = {"--overwrite", "--try-smrf", "--no-html", "--no-consensus"}

    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            if tok in boolean_flags:
                out[key] = True
            elif i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                out[key] = cmd[i + 1]
                i += 1
        i += 1
    return out


def parse_step_from_log_line(line: str) -> tuple[int, str] | None:
    m = re.search(r"Step\s+([0-9]+):\s*(.+)$", line)
    if not m:
        return None
    try:
        step_idx = int(m.group(1))
    except Exception:
        return None
    return step_idx, m.group(2).strip()


def validate_auto_or_numeric(
    value: str,
    *,
    field_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[bool, str | None]:
    s = value.strip()
    if not s:
        return False, f"{field_name}: value is required."

    m = re.fullmatch(r"auto:p([0-9]+(?:\.[0-9]+)?)", s, flags=re.IGNORECASE)
    if m:
        pct = float(m.group(1))
        if pct < 0 or pct > 100:
            return False, f"{field_name}: auto percentile must be between 0 and 100 (e.g., auto:p96)."
        return True, None

    try:
        val = float(s)
    except Exception:
        return False, f"{field_name}: expected numeric value or auto:pNN (e.g., auto:p96)."

    if min_value is not None and val < min_value:
        return False, f"{field_name}: must be >= {min_value}."
    if max_value is not None and val > max_value:
        return False, f"{field_name}: must be <= {max_value}."
    return True, None


def validate_cluster_eps(value: str) -> tuple[bool, str | None]:
    s = value.strip().lower()
    if not s:
        return False, "Cluster radius (m): value is required."
    if s == "auto":
        return True, None
    try:
        v = float(s)
    except Exception:
        return False, "Cluster radius (m): expected 'auto' or a positive number (e.g., 150)."
    if v <= 0:
        return False, "Cluster radius (m): must be > 0 when numeric."
    return True, None


def parse_consensus_percentiles_csv(value: str) -> tuple[list[float] | None, str | None]:
    s = value.strip()
    if not s:
        return None, "Consensus percentiles: value is required (example: 95,96,97)."
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None, "Consensus percentiles: provide comma-separated numbers (example: 95,96,97)."

    vals: list[float] = []
    for p in parts:
        try:
            v = float(p)
        except Exception:
            return None, f"Consensus percentiles: invalid number `{p}`."
        if v < 0.0 or v > 100.0:
            return None, "Consensus percentiles: each value must be between 0 and 100."
        vals.append(v)

    uniq: list[float] = []
    seen: set[float] = set()
    for v in vals:
        key = round(v, 6)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    if not uniq:
        return None, "Consensus percentiles: at least one value is required."
    return uniq, None


def zip_run_dir(run_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(run_dir.parent))
    return zip_path


def read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def wait_for_file(path: Path, timeout_s: float = 6.0, poll_s: float = 0.25) -> bool:
    """
    Small UX helper: pipelines sometimes finish, but the last files land a moment later.
    This avoids “it appeared a minute later” confusion.
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(poll_s)
    return path.exists()


def parse_values_used(process_log: str) -> dict:
    out = {}
    m = re.search(r"Initial regions:\s*([0-9]+)", process_log)
    if m:
        out["regions_initial"] = int(m.group(1))

    m = re.search(r"Filtered regions \(after size \+ region-slope q75\):\s*([0-9]+)", process_log)
    if m:
        out["regions_after_slope"] = int(m.group(1))

    m = re.search(r"Positive relief threshold \(m\):\s*([0-9.]+)\s*\(spec=([^)]+)\)", process_log)
    if m:
        out["pos_thresh_m"] = float(m.group(1))
        out["pos_thresh_spec"] = m.group(2).strip()

    m = re.search(r"Min density threshold:\s*([0-9.]+)\s*\(spec=([^)]+)\)", process_log)
    if m:
        out["min_density"] = float(m.group(1))
        out["min_density_spec"] = m.group(2).strip()

    m = re.search(r"DBSCAN eps auto-chosen:\s*([0-9.]+)\s*m\s*\(min_samples=([0-9]+)\)", process_log)
    if m:
        out["dbscan_eps_m"] = float(m.group(1))
        out["dbscan_min_samples"] = int(m.group(2))

    m = re.search(r"Kept candidates after density \+ post-filters:\s*([0-9]+)", process_log)
    if m:
        out["candidates_kept"] = int(m.group(1))

    m = re.search(r"Dropped by edge buffer \([^)]*\):\s*([0-9]+)", process_log)
    if m:
        out["dropped_edge"] = int(m.group(1))

    m = re.search(r"Dropped by consensus support:\s*([0-9]+)", process_log)
    if m:
        out["dropped_consensus"] = int(m.group(1))

    m = re.search(r"Dropped by density \(region mean < min_density\):\s*([0-9]+)", process_log)
    if m:
        out["dropped_density"] = int(m.group(1))

    m = re.search(r"Dropped by post-filters:\s*([0-9]+)", process_log)
    if m:
        out["dropped_post"] = int(m.group(1))

    m = re.search(r"Dropped by spacing de-dup \([^)]*\):\s*([0-9]+)", process_log)
    if m:
        out["dropped_spacing"] = int(m.group(1))

    m = re.search(r"Clusters found:\s*([0-9]+)\s*\(noise=([0-9]+)\)", process_log)
    if m:
        out["clusters_found"] = int(m.group(1))
        out["clusters_noise"] = int(m.group(2))

    return out


def _to_float_or_none(value) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _to_int_or_none(value) -> int | None:
    try:
        if value is None:
            return None
        v = int(value)
        return v
    except Exception:
        return None


def assess_run_quality(used: dict, df: pd.DataFrame | None) -> dict[str, object]:
    """
    Heuristic quality badge for triage confidence (not ground truth).
    """
    cand_count = _to_int_or_none(used.get("candidates_kept"))
    clusters = _to_int_or_none(used.get("clusters_found"))
    noise = _to_int_or_none(used.get("clusters_noise"))
    top_score = None
    median_score = None
    if df is not None and not df.empty and "score" in df.columns:
        s = pd.to_numeric(df["score"], errors="coerce").dropna()
        if not s.empty:
            top_score = float(s.max())
            median_score = float(s.median())

    noise_frac = None
    if clusters is not None and noise is not None and cand_count and cand_count > 0:
        noise_frac = float(noise) / float(max(1, cand_count))

    check_details: list[dict[str, object]] = [
        {
            "label": "Candidate count in useful review range",
            "target": "8-250",
            "observed": "—" if cand_count is None else str(cand_count),
            "passed": (cand_count is not None and 8 <= cand_count <= 250),
        },
        {
            "label": "At least one non-noise cluster",
            "target": ">= 1 cluster",
            "observed": "—" if clusters is None else str(clusters),
            "passed": (clusters is not None and clusters >= 1),
        },
        {
            "label": "Top score threshold",
            "target": ">= 2.0",
            "observed": "—" if top_score is None else f"{top_score:.3f}",
            "passed": (top_score is not None and top_score >= 2.0),
        },
        {
            "label": "Median score threshold",
            "target": ">= 0.35",
            "observed": "—" if median_score is None else f"{median_score:.3f}",
            "passed": (median_score is not None and median_score >= 0.35),
        },
        {
            "label": "Noise fraction threshold",
            "target": "<= 0.70",
            "observed": "—" if noise_frac is None else f"{noise_frac:.3f}",
            "passed": (None if noise_frac is None else noise_frac <= 0.70),
        },
    ]

    checks = [(str(c["label"]), bool(c["passed"])) for c in check_details if c.get("passed") is not None]

    total = len(checks)
    passed = sum(1 for _, ok in checks if ok)
    ratio = (float(passed) / float(total)) if total else 0.0

    if ratio >= 0.75:
        label = "Strong signal"
        tone = "#0f766e"
        bg = "#ecfdf5"
    elif ratio >= 0.45:
        label = "Moderate signal"
        tone = "#b45309"
        bg = "#fffbeb"
    else:
        label = "Weak / noisy signal"
        tone = "#b91c1c"
        bg = "#fef2f2"

    return {
        "label": label,
        "tone": tone,
        "bg": bg,
        "passed": passed,
        "total": total,
        "checks": checks,
        "check_details": check_details,
        "candidate_count": cand_count,
        "clusters": clusters,
        "noise": noise,
        "top_score": top_score,
        "median_score": median_score,
    }


def quality_action_hints(quality: dict[str, object], used: dict) -> list[str]:
    hints: list[str] = []
    cand_count = _to_int_or_none(quality.get("candidate_count"))
    clusters = _to_int_or_none(quality.get("clusters"))
    top_score = _to_float_or_none(quality.get("top_score"))
    median_score = _to_float_or_none(quality.get("median_score"))
    pos_spec = str(used.get("pos_thresh_spec") or "").strip()
    dens_spec = str(used.get("min_density_spec") or "").strip()

    if cand_count is not None and cand_count < 8:
        hints.append("Candidate count is low. Try `Exploratory` preset or lower `min-density` and `min-prominence` slightly.")
    if clusters is not None and clusters < 1:
        hints.append("No clusters found. Try `--cluster-eps auto` (or increase fixed eps) and reduce `min-samples` to 3-4.")
    if top_score is not None and top_score < 2.0:
        hints.append("Top score is low. Relax shape strictness (`min_compactness`, `min_solidity`, `min_extent`) a bit.")
    if median_score is not None and median_score < 0.35:
        hints.append("Median score is low. Raise relief signal by lowering `pos-thresh` percentile (e.g., `auto:p95`).")
    if pos_spec and pos_spec.startswith("auto:p") and pos_spec.lower() not in {"auto:p95", "auto:p94"}:
        hints.append("For noisy/flat tiles, test `pos-thresh auto:p95` and compare outcomes in Preset Comparison.")
    if dens_spec and dens_spec.startswith("auto:p") and dens_spec.lower() not in {"auto:p55", "auto:p50"}:
        hints.append("If density gate is strict, try `min-density auto:p55` and inspect filter waterfall deltas.")

    return hints[:4]


def tuning_hints_from_filter_waterfall(
    dropped_edge: int | None,
    dropped_consensus: int | None,
    dropped_density: int | None,
    dropped_post: int | None,
    dropped_spacing: int | None,
    kept: int | None,
) -> list[str]:
    vals = {
        "edge": int(dropped_edge or 0),
        "consensus": int(dropped_consensus or 0),
        "density": int(dropped_density or 0),
        "post": int(dropped_post or 0),
        "spacing": int(dropped_spacing or 0),
        "kept": int(kept or 0),
    }
    total = sum(vals.values())
    if total <= 0:
        return []

    hints: list[str] = []
    kept_count = vals["kept"]
    if kept_count < 8:
        hints.append("Recall is low. Try easing strictness: lower `min_density`, lower `min_prominence`, or reduce `consensus_min_support`.")
    elif kept_count > 300:
        hints.append("Candidate count is very high. Tighten strictness: raise `pos-thresh`, raise `min_density`, or increase `min_compactness`.")

    if vals["consensus"] / total >= 0.35:
        hints.append("Consensus is dropping many regions. Consider a larger `consensus-radius-m` or lower `consensus-min-support`.")
    if vals["density"] / total >= 0.35:
        hints.append("Density gate dominates drops. Lower `min-density` percentile or reduce `density-sigma` for less neighborhood smoothing.")
    if vals["post"] / total >= 0.45:
        hints.append("Shape/physics filters dominate. Recheck `min_peak`, `min_extent`, `min_compactness`, `min_solidity`, and area bounds.")
    if vals["spacing"] / total >= 0.30:
        hints.append("Spacing de-dup is collapsing nearby points. Lower `min-spacing-m` if you need finer settlement granularity.")
    if vals["edge"] / total >= 0.30:
        hints.append("Edge buffer is dropping many regions. Reduce `edge-buffer-m` or process larger tiles with overlap.")

    return hints[:4]


def score_formula_strings(run_params_data: dict | None) -> tuple[dict[str, float], str, str, str]:
    params = {}
    if isinstance(run_params_data, dict):
        p = run_params_data.get("params")
        if isinstance(p, dict):
            params = p

    exps = {
        "density": float(params.get("score_density_exp", 1.0)),
        "peak_relief_m": float(params.get("score_peak_exp", 1.0)),
        "extent": float(params.get("score_extent_exp", 0.35)),
        "consensus_support": float(params.get("score_consensus_exp", 0.40)),
        "prominence_m": float(params.get("score_prominence_exp", 0.75)),
        "compactness": float(params.get("score_compactness_exp", 0.20)),
        "solidity": float(params.get("score_solidity_exp", 0.20)),
        "area_m2": float(params.get("score_area_exp", 0.50)),
    }
    a = exps["density"]
    b = exps["peak_relief_m"]
    c = exps["extent"]
    d = exps["consensus_support"]
    e = exps["prominence_m"]
    f = exps["compactness"]
    g = exps["solidity"]
    h = exps["area_m2"]

    formula_generic = (
        "score = density^a * peak^b * extent^c * consensus_support^d * "
        "prominence^e * compactness^f * solidity^g * area^h"
    )
    formula_numeric = (
        "score = density^"
        f"{a:.2f} * peak^{b:.2f} * extent^{c:.2f} * consensus_support^{d:.2f} * "
        f"prominence^{e:.2f} * compactness^{f:.2f} * solidity^{g:.2f} * area^{h:.2f}"
    )
    exponent_legend = (
        f"a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}, "
        f"e={e:.2f}, f={f:.2f}, g={g:.2f}, h={h:.2f}"
    )
    return exps, formula_generic, formula_numeric, exponent_legend


def score_formula_readable(run_params_data: dict | None) -> tuple[str, pd.DataFrame]:
    exps, _, _, _ = score_formula_strings(run_params_data)
    readable = (
        "Score =\n"
        f"  Density^{exps['density']:.2f}\n"
        f"  x PeakRelief^{exps['peak_relief_m']:.2f}\n"
        f"  x Extent^{exps['extent']:.2f}\n"
        f"  x Support^{exps['consensus_support']:.2f}\n"
        f"  x Prominence^{exps['prominence_m']:.2f}\n"
        f"  x Compactness^{exps['compactness']:.2f}\n"
        f"  x Solidity^{exps['solidity']:.2f}\n"
        f"  x Area^{exps['area_m2']:.2f}"
    )
    rows = [
        {
            "Component": "Density",
            "Exponent": round(exps["density"], 2),
            "Meaning": "How concentrated nearby candidate features are.",
        },
        {
            "Component": "Peak Relief",
            "Exponent": round(exps["peak_relief_m"], 2),
            "Meaning": "Strength of the local topographic high point.",
        },
        {
            "Component": "Extent",
            "Exponent": round(exps["extent"], 2),
            "Meaning": "How much of the candidate bounding box is filled.",
        },
        {
            "Component": "Consensus Support",
            "Exponent": round(exps["consensus_support"], 2),
            "Meaning": "How consistently the candidate appears across thresholds.",
        },
        {
            "Component": "Prominence",
            "Exponent": round(exps["prominence_m"], 2),
            "Meaning": "How much the candidate stands out from nearby terrain.",
        },
        {
            "Component": "Compactness",
            "Exponent": round(exps["compactness"], 2),
            "Meaning": "How non-line-like the shape is.",
        },
        {
            "Component": "Solidity",
            "Exponent": round(exps["solidity"], 2),
            "Meaning": "How solid/convex (vs fragmented) the shape is.",
        },
        {
            "Component": "Area",
            "Exponent": round(exps["area_m2"], 2),
            "Meaning": "Footprint size influence in the final ranking.",
        },
    ]
    return readable, pd.DataFrame(rows)


def score_breakdown_df(row: pd.Series, run_params_data: dict | None) -> pd.DataFrame | None:
    exps, _, _, _ = score_formula_strings(run_params_data)
    floors = {
        "density": 1e-9,
        "peak_relief_m": 1e-9,
        "extent": 1e-6,
        "consensus_support": 1.0,
        "prominence_m": 1e-6,
        "compactness": 1e-6,
        "solidity": 1e-6,
        "area_m2": 1e-9,
    }
    label_map = {
        "density": "density",
        "peak_relief_m": "peak relief (m)",
        "extent": "extent",
        "consensus_support": "consensus support",
        "prominence_m": "prominence (m)",
        "compactness": "compactness",
        "solidity": "solidity",
        "area_m2": "area (m²)",
    }

    rows = []
    est_score = 1.0
    for key in ["density", "peak_relief_m", "extent", "consensus_support", "prominence_m", "compactness", "solidity", "area_m2"]:
        if key not in row:
            return None
        raw = _to_float_or_none(row.get(key))
        if raw is None:
            return None
        raw_f = max(float(floors[key]), float(raw))
        exp = float(exps[key])
        term = float(raw_f ** exp)
        est_score *= term
        rows.append(
            {
                "component": label_map[key],
                "raw_value": raw,
                "exponent": exp,
                "term": term,
            }
        )

    out = pd.DataFrame(rows)
    out["term_norm"] = out["term"] / max(1e-12, float(out["term"].max()))
    out.attrs["estimated_score"] = float(est_score)
    return out


def build_provenance_text(
    *,
    run_dir: Path,
    command: list[str] | None,
    used: dict,
    run_params_data: dict | None,
    preset_name: str | None,
    preset_match: bool | None,
) -> str:
    lines: list[str] = []
    lines.append(f"run_dir={run_dir}")
    if run_params_data and isinstance(run_params_data, dict):
        ts = run_params_data.get("timestamp_utc")
        if ts:
            lines.append(f"timestamp_utc={ts}")
        input_path = run_params_data.get("input_path")
        if input_path:
            lines.append(f"input={input_path}")
    lines.append(f"ui_preset_selected={preset_name or 'unknown'}")
    lines.append(f"ui_preset_match={preset_match}")

    if used:
        lines.append(f"resolved_pos_thresh_m={used.get('pos_thresh_m')}")
        lines.append(f"resolved_pos_thresh_spec={used.get('pos_thresh_spec')}")
        lines.append(f"resolved_min_density={used.get('min_density')}")
        lines.append(f"resolved_min_density_spec={used.get('min_density_spec')}")
        lines.append(f"resolved_dbscan_eps_m={used.get('dbscan_eps_m')}")
        lines.append(f"resolved_dbscan_min_samples={used.get('dbscan_min_samples')}")
        lines.append(f"resolved_candidates_kept={used.get('candidates_kept')}")
        lines.append(f"resolved_clusters_found={used.get('clusters_found')}")
        lines.append(f"resolved_clusters_noise={used.get('clusters_noise')}")
        lines.append(f"resolved_dropped_edge={used.get('dropped_edge')}")
        lines.append(f"resolved_dropped_density={used.get('dropped_density')}")
        lines.append(f"resolved_dropped_post={used.get('dropped_post')}")
        lines.append(f"resolved_dropped_spacing={used.get('dropped_spacing')}")

    if run_params_data and isinstance(run_params_data, dict):
        params = run_params_data.get("params")
        if isinstance(params, dict):
            for k in sorted(params.keys()):
                lines.append(f"param.{k}={params[k]}")

    if command:
        lines.append("command=" + " ".join(command))

    return "\n".join(lines).strip() + "\n"


def render_hint_block(title: str, hints: list[str], *, level: str = "info") -> None:
    if level == "warning":
        st.warning(title)
    elif level == "error":
        st.error(title)
    else:
        st.info(title)
    st.markdown("\n".join([f"- {h}" for h in hints]))


def run_failure_hints(log_text: str) -> list[str]:
    t = (log_text or "").lower()
    hints = [
        "Open the **Run details** tab and check the last error line in logs.",
        "Try a new run name or enable `Overwrite run folder`.",
    ]
    if "pdal" in t and ("not found" in t or "no such file" in t):
        hints.insert(0, "PDAL appears unavailable; install/verify it (`pdal --version`).")
    if "input laz/las not found" in t or "input file not found" in t:
        hints.insert(0, "Input path looks invalid; verify the selected `.laz/.las` file exists.")
    if "run dir already exists" in t:
        hints.insert(0, "Run folder already exists; enable overwrite or change run name.")
    if "dtm has no crs" in t:
        hints.insert(0, "DTM CRS is missing; confirm source LAS/LAZ has valid georeferencing.")
    return hints


@st.cache_data(show_spinner=False)
def inline_report_images_and_basemap(report_html_path_str: str, run_dir_str: str, report_mtime_ns: int) -> str:
    """
    1) Inline html/img/*.png as data URIs so embedded report shows images.
    2) Add Street/Satellite basemap toggle to the Leaflet map in report.html (if applicable).
    """
    # report_mtime_ns is included for cache invalidation when report.html changes
    _ = report_mtime_ns
    report_html_path = Path(report_html_path_str)
    run_dir = Path(run_dir_str)
    html = read_text_safely(report_html_path)
    if not html.strip():
        return ""

    # Inline local PNGs
    pattern = re.compile(r"""src=(['"])(html/img/[^'"]+)\1""", re.IGNORECASE)

    def repl(match):
        rel = match.group(2)
        img_path = run_dir / rel
        if not img_path.exists():
            return match.group(0)
        try:
            b = img_path.read_bytes()
            b64 = base64.b64encode(b).decode("ascii")
            return f'''src="data:image/png;base64,{b64}"'''
        except Exception:
            return match.group(0)

    html2 = pattern.sub(repl, html)

    # Make embedded map taller
    html2 = html2.replace("#map { height: 520px;", "#map { height: 680px;")

    # Replace the single OSM layer with a Street/Satellite basemap control (best-effort)
    tilelayer_re = re.compile(
        r"""L\.tileLayer\(\s*['"]https://\{s\}\.tile\.openstreetmap\.org/\{z\}/\{x\}/\{y\}\.png['"]\s*,\s*\{.*?\}\s*\)\.addTo\(map\);""",
        re.DOTALL,
    )

    replacement = """
// Basemaps
const street = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
});

const satellite = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { maxZoom: 19, attribution: 'Tiles &copy; Esri' }
);

// Default basemap
street.addTo(map);

// Layer control
L.control.layers(
  { "Street": street, "Satellite": satellite },
  {},
  { collapsed: true, position: 'topright' }
).addTo(map);
""".strip()

    if tilelayer_re.search(html2):
        html2 = tilelayer_re.sub(replacement, html2)

    # Align report-tab map framing with Results map tab:
    # fit all candidates with adaptive padding/zoom, plus single-point fallback.
    fit_bounds_re = re.compile(
        r"""if\s*\(\s*bounds\.length\s*>\s*0\s*\)\s*\{\s*map\.fitBounds\(\s*bounds\s*,\s*\{\s*padding\s*:\s*\[[^\]]+\]\s*\}\s*\)\s*;\s*\}""",
        re.DOTALL,
    )

    fit_bounds_replacement = """
if (bounds.length > 1) {
  const latLngBounds = L.latLngBounds(bounds);
  const lats = bounds.map(b => Number(b[0]));
  const lons = bounds.map(b => Number(b[1]));
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const meanLatRad = ((minLat + maxLat) / 2.0) * Math.PI / 180.0;

  // Approximate bbox perimeter in kilometers to tune initial zoom/padding.
  const latKm = Math.abs(maxLat - minLat) * 111.32;
  const lonKm = Math.abs(maxLon - minLon) * 111.32 * Math.max(0.1, Math.cos(meanLatRad));
  const perimeterKm = 2.0 * (latKm + lonKm);

  let padFrac = 0.04;
  let maxZoom = 15;
  if (perimeterKm >= 150) {
    padFrac = 0.12;
    maxZoom = 11;
  } else if (perimeterKm >= 80) {
    padFrac = 0.09;
    maxZoom = 12;
  } else if (perimeterKm >= 35) {
    padFrac = 0.07;
    maxZoom = 13;
  } else if (perimeterKm >= 15) {
    padFrac = 0.05;
    maxZoom = 14;
  } else if (perimeterKm >= 6) {
    padFrac = 0.04;
    maxZoom = 15;
  } else if (perimeterKm >= 2) {
    padFrac = 0.03;
    maxZoom = 16;
  } else {
    padFrac = 0.02;
    maxZoom = 17;
  }

  map.fitBounds(latLngBounds.pad(padFrac), { padding: [12, 12], maxZoom: maxZoom });
} else if (bounds.length === 1) {
  // Single-point runs should still open with context around the candidate.
  map.setView(bounds[0], 16);
}
""".strip()

    if fit_bounds_re.search(html2):
        html2 = fit_bounds_re.sub(fit_bounds_replacement, html2)

    return html2


@st.cache_data(show_spinner=False)
def _load_candidates_cached(csv_path_str: str, csv_mtime_ns: int) -> pd.DataFrame | None:
    # csv_mtime_ns is included for cache invalidation when candidates.csv changes
    _ = csv_mtime_ns
    p = Path(csv_path_str)
    try:
        df = pd.read_csv(p)
        if "lat" not in df.columns or "lon" not in df.columns:
            return None
        df = df.dropna(subset=["lat", "lon"]).copy()
        if "cluster_id" in df.columns:
            df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
        else:
            df["cluster_id"] = -1
        return df
    except Exception:
        return None


def load_candidates(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "candidates.csv"
    if not p.exists():
        return None
    try:
        return _load_candidates_cached(str(p), p.stat().st_mtime_ns)
    except Exception:
        return None


def labels_path_for_run(run_dir: Path) -> Path:
    return run_dir / "candidate_labels.csv"


def load_candidate_labels(run_dir: Path) -> pd.DataFrame:
    path = labels_path_for_run(run_dir)
    if not path.exists():
        return pd.DataFrame(columns=["cand_id", "label", "note", "updated_utc", "lat", "lon", "score"])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["cand_id", "label", "note", "updated_utc", "lat", "lon", "score"])

    for col in ["cand_id", "label", "note", "updated_utc", "lat", "lon", "score"]:
        if col not in df.columns:
            df[col] = None
    df["cand_id"] = pd.to_numeric(df["cand_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cand_id"]).copy()
    df["cand_id"] = df["cand_id"].astype(int)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    valid = {"likely", "unlikely", "unknown"}
    df = df[df["label"].isin(valid)].copy()
    return df[["cand_id", "label", "note", "updated_utc", "lat", "lon", "score"]]


def upsert_candidate_label(
    run_dir: Path,
    cand_id: int,
    label: str,
    note: str,
    lat: float | None = None,
    lon: float | None = None,
    score: float | None = None,
) -> Path:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    clean_label = str(label).strip().lower()
    if clean_label not in {"likely", "unlikely", "unknown"}:
        clean_label = "unknown"
    clean_note = str(note).strip()

    df = load_candidate_labels(run_dir)
    row = {
        "cand_id": int(cand_id),
        "label": clean_label,
        "note": clean_note,
        "updated_utc": now_utc,
        "lat": lat,
        "lon": lon,
        "score": score,
    }

    if not df.empty and int(cand_id) in set(df["cand_id"].astype(int).tolist()):
        df.loc[df["cand_id"].astype(int) == int(cand_id), list(row.keys())] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    path = labels_path_for_run(run_dir)
    df.sort_values(["cand_id"], inplace=True)
    df.to_csv(path, index=False)
    return path


def merge_labels_into_candidates(cands_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    out = cands_df.copy()
    out["cand_id"] = pd.to_numeric(out["cand_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["cand_id"]).copy()
    out["cand_id"] = out["cand_id"].astype(int)
    if labels_df is None or labels_df.empty:
        out["analyst_label"] = "unknown"
        out["analyst_note"] = ""
        out["analyst_updated_utc"] = ""
        return out

    lbl = labels_df.copy()
    lbl["cand_id"] = pd.to_numeric(lbl["cand_id"], errors="coerce").astype("Int64")
    lbl = lbl.dropna(subset=["cand_id"]).copy()
    lbl["cand_id"] = lbl["cand_id"].astype(int)
    lbl = lbl.drop_duplicates(subset=["cand_id"], keep="last")
    lbl = lbl.rename(
        columns={
            "label": "analyst_label",
            "note": "analyst_note",
            "updated_utc": "analyst_updated_utc",
        }
    )
    out = out.merge(
        lbl[["cand_id", "analyst_label", "analyst_note", "analyst_updated_utc"]],
        on="cand_id",
        how="left",
    )
    out["analyst_label"] = out["analyst_label"].fillna("unknown")
    out["analyst_note"] = out["analyst_note"].fillna("")
    out["analyst_updated_utc"] = out["analyst_updated_utc"].fillna("")
    return out


def candidate_label_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or "analyst_label" not in df.columns:
        return {}

    mdf = df.copy()
    mdf["analyst_label"] = mdf["analyst_label"].astype(str).str.lower()
    labeled = mdf[mdf["analyst_label"].isin(["likely", "unlikely"])].copy()
    likely_n = int((mdf["analyst_label"] == "likely").sum())
    unlikely_n = int((mdf["analyst_label"] == "unlikely").sum())
    unknown_n = int((mdf["analyst_label"] == "unknown").sum())

    metrics: Dict[str, Any] = {
        "labeled_total": int(len(labeled)),
        "likely_count": likely_n,
        "unlikely_count": unlikely_n,
        "unknown_count": unknown_n,
    }

    denom = max(1, likely_n + unlikely_n)
    if likely_n + unlikely_n > 0:
        metrics["analyst_precision"] = float(likely_n / denom)
        metrics["analyst_false_positive_rate"] = float(unlikely_n / denom)

    if "score" in mdf.columns:
        mdf["score"] = pd.to_numeric(mdf["score"], errors="coerce")
        mdf = mdf.sort_values("score", ascending=False)
        p_at_k: Dict[str, float] = {}
        coverage_at_k: Dict[str, float] = {}
        for k in (5, 10, 20):
            topk = mdf.head(min(k, len(mdf))).copy()
            if topk.empty:
                continue
            topk_labeled = topk[topk["analyst_label"].isin(["likely", "unlikely"])]
            if topk_labeled.empty:
                continue
            p_at_k[f"p_at_{k}"] = float((topk_labeled["analyst_label"] == "likely").mean())
            coverage_at_k[f"coverage_at_{k}"] = float(len(topk_labeled) / max(1, len(topk)))
        metrics.update(p_at_k)
        metrics.update(coverage_at_k)
    return metrics


# -----------------------------
# Leaflet map (no Mapbox token needed)
# -----------------------------
def leaflet_map_html(df: pd.DataFrame) -> str:
    """
    A reliable, no-API-key map:
    - Street + Satellite basemap toggle
    - Fits to all candidate points
    - Click a point to see key metrics
    """
    # Keep only what we need (prevents gigantic HTML)
    cols = [c for c in ["cand_id", "score", "peak_relief_m", "consensus_support", "prominence_m", "area_m2", "extent", "aspect", "compactness", "solidity", "cluster_id", "analyst_label", "lat", "lon"] if c in df.columns]
    pts = df[cols].copy()
    has_analyst_label = "analyst_label" in pts.columns

    # Ensure numeric for JS formatting
    for c in ["score", "peak_relief_m", "consensus_support", "prominence_m", "area_m2", "extent", "aspect", "compactness", "solidity", "lat", "lon"]:
        if c in pts.columns:
            pts[c] = pd.to_numeric(pts[c], errors="coerce")

    pts = pts.dropna(subset=["lat", "lon"]).to_dict(orient="records")
    data_json = json.dumps(pts)

    return f"""
<div id="map" style="height:720px; border-radius:12px; overflow:hidden;"></div>

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const data = {data_json};

const map = L.map('map', {{
  zoomControl: true
}});

const street = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}});

const satellite = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri'
  }}
);

street.addTo(map);
L.control.layers({{ "Street": street, "Satellite": satellite }}, {{}}, {{ collapsed: true, position: 'topright' }}).addTo(map);

const bounds = [];
const clusterCounts = {{}};
const palette = [
  "#c0392b", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
  "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f"
];

function fmt(v, digits=3) {{
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}}

function clusterColor(cid) {{
  if (cid === -1) return "#666";
  const idx = Math.abs(cid - 1) % palette.length;
  return palette[idx];
}}

data.forEach(p => {{
  const lat = Number(p.lat);
  const lon = Number(p.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

  const cid = (p.cluster_id === null || p.cluster_id === undefined) ? -1 : Number(p.cluster_id);
  const color = clusterColor(cid);
  clusterCounts[cid] = (clusterCounts[cid] || 0) + 1;

  const marker = L.circleMarker([lat, lon], {{
    radius: 6,
    color: color,
    weight: 2,
    fillOpacity: 0.7
  }});

    const popup = `
      <b>Candidate ${"{p.cand_id}"}</b><br/>
      Score: ${"{fmt(p.score, 3)}"}<br/>
      Peak relief (m): ${"{fmt(p.peak_relief_m, 2)}"}<br/>
      Consensus support: ${"{fmt(p.consensus_support, 0)}"}<br/>
      Prominence (m): ${"{fmt(p.prominence_m, 2)}"}<br/>
      Area (m²): ${"{fmt(p.area_m2, 1)}"}<br/>
      Extent: ${"{fmt(p.extent, 3)}"}<br/>
      Compactness: ${"{fmt(p.compactness, 3)}"}<br/>
      Solidity: ${"{fmt(p.solidity, 3)}"}<br/>
      Elongation: ${"{fmt(p.aspect, 2)}"}<br/>
      Cluster: ${"{cid}"}<br/>
      {"Analyst label: ${p.analyst_label ?? 'unknown'}" if has_analyst_label else ""}
    `;

  marker.bindPopup(popup);
  marker.addTo(map);
  bounds.push([lat, lon]);
}});

if (Object.keys(clusterCounts).length > 0) {{
  const legend = L.control({{ position: "bottomright" }});
  legend.onAdd = function() {{
    const div = L.DomUtil.create("div");
    div.style.background = "rgba(255,255,255,0.92)";
    div.style.padding = "8px 10px";
    div.style.border = "1px solid #ddd";
    div.style.borderRadius = "8px";
    div.style.boxShadow = "0 1px 4px rgba(0,0,0,0.15)";
    div.style.fontSize = "12px";
    div.style.lineHeight = "1.3";
    div.style.maxHeight = "180px";
    div.style.overflowY = "auto";

    const ids = Object.keys(clusterCounts)
      .map(v => Number(v))
      .sort((a, b) => (a === -1 ? -999 : a) - (b === -1 ? -999 : b));

    let rows = '<div style="font-weight:600; margin-bottom:6px;">Clusters</div>';
    ids.forEach(cid => {{
      const color = clusterColor(cid);
      const label = (cid === -1) ? "Noise (-1)" : "Cluster " + cid;
      const count = clusterCounts[cid] || 0;
      rows += '<div style="display:flex; align-items:center; margin:2px 0;">'
        + '<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:' + color + '; margin-right:8px;"></span>'
        + '<span>' + label + " (" + count + ")" + "</span>"
        + "</div>";
    }});

    div.innerHTML = rows;
    return div;
  }};
  legend.addTo(map);
}}

if (bounds.length > 1) {{
  const latLngBounds = L.latLngBounds(bounds);
  const lats = bounds.map(b => Number(b[0]));
  const lons = bounds.map(b => Number(b[1]));
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const meanLatRad = ((minLat + maxLat) / 2.0) * Math.PI / 180.0;

  // Approximate bbox perimeter in kilometers to tune initial zoom/padding.
  const latKm = Math.abs(maxLat - minLat) * 111.32;
  const lonKm = Math.abs(maxLon - minLon) * 111.32 * Math.max(0.1, Math.cos(meanLatRad));
  const perimeterKm = 2.0 * (latKm + lonKm);

  let padFrac = 0.04;
  let maxZoom = 15;
  if (perimeterKm >= 150) {{
    padFrac = 0.12;
    maxZoom = 11;
  }} else if (perimeterKm >= 80) {{
    padFrac = 0.09;
    maxZoom = 12;
  }} else if (perimeterKm >= 35) {{
    padFrac = 0.07;
    maxZoom = 13;
  }} else if (perimeterKm >= 15) {{
    padFrac = 0.05;
    maxZoom = 14;
  }} else if (perimeterKm >= 6) {{
    padFrac = 0.04;
    maxZoom = 15;
  }} else if (perimeterKm >= 2) {{
    padFrac = 0.03;
    maxZoom = 16;
  }} else {{
    padFrac = 0.02;
    maxZoom = 17;
  }}

  map.fitBounds(latLngBounds.pad(padFrac), {{ padding: [12, 12], maxZoom: maxZoom }});
}} else if (bounds.length === 1) {{
  // Single-point runs should still open with context around the candidate.
  map.setView(bounds[0], 16);
}} else {{
  map.setView([17.0, -89.0], 10);
}}
</script>
"""


# -----------------------------
# Page chrome
# -----------------------------
st.set_page_config(page_title="MayaScan", layout="wide")

st.markdown(
    """
<style>
/* Make sure header never looks clipped across browser chrome/zoom variants */
.block-container { padding-top: clamp(2.0rem, 3vh, 2.4rem) !important; padding-bottom: 2rem; }

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem !important; }

/* Reduce widget gaps slightly */
div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }

/* Round code blocks */
pre { border-radius: 12px !important; }

/* Tabs padding */
button[data-baseweb="tab"] { padding-top: 8px; padding-bottom: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
if LOGO_PATH.exists():
    try:
        logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        header_icon_html = (
            f"<img src='data:image/svg+xml;base64,{logo_b64}' "
            "alt='MayaScan logo' style='width:78px; height:78px; display:block;'/>"
        )
    except Exception:
        header_icon_html = "<div style='font-size:30px; line-height:1;'>🏛️</div>"
else:
    header_icon_html = "<div style='font-size:30px; line-height:1;'>🏛️</div>"

st.markdown(
    f"""
<div style="display:flex; align-items:center; gap:12px; margin:0.35rem 0 0.35rem 0;">
  <div>{header_icon_html}</div>
  <div>
    <div style="font-size:30px; font-weight:800; line-height:1.15;">MayaScan</div>
    <div style="color:#555; font-size:14px; margin-top:3px;">
      Upload a LiDAR tile, run the pipeline, and review results (map + ranked candidates).
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if not SCRIPT_PATH.exists():
    st.error(f"Could not find maya_scan.py at: {SCRIPT_PATH}")
    st.stop()

if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_cmd" not in st.session_state:
    st.session_state.last_cmd = None
if "last_logs" not in st.session_state:
    st.session_state.last_logs = ""
if "zip_ready_for_run" not in st.session_state:
    st.session_state.zip_ready_for_run = None
if "zip_path" not in st.session_state:
    st.session_state.zip_path = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "last_preset_selected" not in st.session_state:
    st.session_state.last_preset_selected = None
if "last_preset_match" not in st.session_state:
    st.session_state.last_preset_match = None
if "last_ui_config" not in st.session_state:
    st.session_state.last_ui_config = None
if "last_compare_summary" not in st.session_state:
    st.session_state.last_compare_summary = None
if "cfg_portfolio_mode" not in st.session_state:
    st.session_state.cfg_portfolio_mode = False
if "cfg_annotation_mode" not in st.session_state:
    st.session_state.cfg_annotation_mode = False
if "cfg_run_name" not in st.session_state:
    st.session_state.cfg_run_name = next_default_run_name()
if "pending_cfg_run_name" not in st.session_state:
    st.session_state.pending_cfg_run_name = None
pending_run_name = st.session_state.get("pending_cfg_run_name")
if isinstance(pending_run_name, str) and pending_run_name.strip():
    st.session_state.cfg_run_name = sanitize_run_name(pending_run_name)
    st.session_state.pending_cfg_run_name = None
if "cfg_compare_presets" not in st.session_state:
    st.session_state.cfg_compare_presets = [PRESET_BALANCED, PRESET_STRICT]
if not isinstance(st.session_state.cfg_compare_presets, list):
    st.session_state.cfg_compare_presets = [PRESET_BALANCED, PRESET_STRICT]
else:
    st.session_state.cfg_compare_presets = normalize_compare_presets(st.session_state.cfg_compare_presets)
    if not st.session_state.cfg_compare_presets:
        st.session_state.cfg_compare_presets = [PRESET_BALANCED, PRESET_STRICT]

if "cfg_preset" not in st.session_state:
    st.session_state.cfg_preset = PRESET_BALANCED
if "cfg_pos_thresh" not in st.session_state:
    apply_preset_to_session(st.session_state.cfg_preset)
if "cfg_preset_prev" not in st.session_state:
    st.session_state.cfg_preset_prev = st.session_state.cfg_preset
for _k, _v in PRESET_VALUES.get(st.session_state.cfg_preset, PRESET_VALUES[PRESET_BALANCED]).items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# -----------------------------
# Sidebar: Inputs + Parameters
# -----------------------------
with st.sidebar:
    st.markdown("### 1) Input")
    mode = st.radio("Source", ["Upload .laz/.las", "Use local path"], horizontal=False)

    uploaded = None
    input_local = ""
    if mode == "Upload .laz/.las":
        uploaded = st.file_uploader("LAZ/LAS file", type=["laz", "las"])
        st.caption("Saved into `data/lidar/` for this run.")
    else:
        input_local = st.text_input("Local path", value="data/lidar/sample.laz")

    st.divider()
    st.markdown("### 2) Run")
    run_name = st.text_input("Run name", key="cfg_run_name")
    runs_dir = st.text_input("Runs directory", value="runs")
    runs_dir_preview = resolve_runs_dir(runs_dir)

    existing_runs = list_existing_runs(runs_dir_preview)
    run_option_labels = {name: summarize_run_option(runs_dir_preview / name) for name in existing_runs}
    selected_existing_run = st.selectbox(
        "Load existing run",
        options=["(none)"] + existing_runs,
        index=0,
        format_func=lambda name: "(none)" if name == "(none)" else run_option_labels.get(name, name),
        help="Load outputs from a previous run without rerunning the pipeline.",
    )
    if st.button(
        "Load selected run",
        disabled=(selected_existing_run == "(none)" or st.session_state.is_running),
    ):
        loaded_dir = runs_dir_preview / selected_existing_run
        st.session_state.last_run_dir = str(loaded_dir)
        st.session_state.last_cmd = None
        st.session_state.last_logs = ""
        st.session_state.last_preset_selected = None
        st.session_state.last_preset_match = None
        st.session_state.last_ui_config = None
        st.session_state.last_compare_summary = None
        st.session_state.zip_ready_for_run = None
        st.session_state.zip_path = None
        st.success(f"Loaded run: {selected_existing_run}")

    overwrite = st.checkbox(
        "Overwrite run folder",
        value=False,
        help=(
            "If enabled, an existing run folder with the same name is deleted and recreated. "
            "If disabled, MayaScan now auto-generates a unique run name on collision."
        ),
    )
    try_smrf = st.checkbox(
        "Ground classification (SMRF)",
        value=True,
        help="Attempts to classify ground points before building the terrain model (DTM). Can help in vegetation-heavy areas.",
    )
    if try_smrf:
        with st.expander("SMRF tuning (advanced)", expanded=False):
            smrf_scalar = st.number_input(
                "SMRF scalar",
                min_value=0.1, max_value=10.0,
                value=st.session_state.get("cfg_smrf_scalar", 1.25),
                step=0.05, key="cfg_smrf_scalar",
                help="Controls the height threshold scaling. Increase in areas with taller vegetation.",
            )
            smrf_slope = st.number_input(
                "SMRF slope",
                min_value=0.01, max_value=1.0,
                value=st.session_state.get("cfg_smrf_slope", 0.15),
                step=0.01, key="cfg_smrf_slope",
                help="Slope filter for ground classification. Lower = stricter on steep terrain.",
            )
            smrf_threshold = st.number_input(
                "SMRF threshold (m)",
                min_value=0.05, max_value=5.0,
                value=st.session_state.get("cfg_smrf_threshold", 0.5),
                step=0.05, key="cfg_smrf_threshold",
                help="Height above ground threshold for ground/non-ground classification.",
            )
            smrf_window = st.number_input(
                "SMRF window (m)",
                min_value=1.0, max_value=100.0,
                value=st.session_state.get("cfg_smrf_window", 16.0),
                step=1.0, key="cfg_smrf_window",
                help="Moving window size. Larger values help in areas with broad vegetation.",
            )
    else:
        smrf_scalar = st.session_state.get("cfg_smrf_scalar", 1.25)
        smrf_slope = st.session_state.get("cfg_smrf_slope", 0.15)
        smrf_threshold = st.session_state.get("cfg_smrf_threshold", 0.5)
        smrf_window = st.session_state.get("cfg_smrf_window", 16.0)

    resolution_m = st.number_input(
        "DTM resolution (m/pixel)",
        min_value=0.25, max_value=10.0,
        value=st.session_state.get("cfg_resolution_m", 1.0),
        step=0.25, key="cfg_resolution_m",
        help=(
            "Ground model pixel size. Default 1.0 m works well for most Caracol-density tiles. "
            "Increase to 2–5 m only for very sparse point clouds (<2 pts/m²)."
        ),
    )
    no_html = st.checkbox(
        "Skip HTML report",
        value=False,
        help="If enabled, MayaScan will not generate report.html or cutout image panels.",
    )
    portfolio_mode = st.toggle(
        "Portfolio mode (clean demo view)",
        key="cfg_portfolio_mode",
        help="Hides diagnostics-heavy sections and keeps presentation-focused outputs prominent.",
    )
    annotation_mode = st.toggle(
        "Annotation mode (manual labels)",
        key="cfg_annotation_mode",
        help="Shows analyst-label filters and candidate labeling UI. Keep off for a cleaner review-focused workflow.",
    )

    st.divider()
    st.markdown("### 3) Scientific preset")
    preset_name = st.selectbox(
        "Parameter profile",
        options=list(PRESET_VALUES.keys()),
        key="cfg_preset",
        help="Use presets for reproducible runs, then fine-tune any field below if needed.",
    )
    preset_changed = preset_name != st.session_state.get("cfg_preset_prev")
    if preset_changed:
        apply_preset_to_session(preset_name, update_selector=False)
        st.session_state.cfg_preset_prev = preset_name
        st.success(f"Applied preset automatically: {preset_name}")

    apply_preset_clicked = st.button("Re-apply preset values", width="stretch")
    if apply_preset_clicked:
        apply_preset_to_session(preset_name, update_selector=False)
        st.session_state.cfg_preset_prev = preset_name
        st.success(f"Re-applied preset: {preset_name}")
    preset_match = preset_matches_session(preset_name)
    if preset_match:
        st.caption("Current values match the selected preset.")
    else:
        st.caption("Current values are custom (modified from preset).")

    st.divider()
    st.markdown("### 4) Detection (what becomes a candidate?)")
    pos_thresh = st.text_input(
        "Relief threshold (auto percentile or meters)",
        key="cfg_pos_thresh",
        help=(
            "Controls how strong a terrain bump must be (in the Local Relief Model) to become a candidate.\n\n"
            "Examples:\n"
            "- `auto:p96` = use the 96th percentile of positive relief values in this tile\n"
            "- `0.25` = fixed 0.25 m relief threshold"
        ),
    )
    min_density = st.text_input(
        "Neighborhood density threshold (auto percentile or 0–1)",
        key="cfg_min_density",
        help=(
            "After candidates are detected, MayaScan builds a smoothed “feature density” raster.\n"
            "This removes isolated noise and keeps candidates in locally feature-rich zones.\n\n"
            "Examples:\n"
            "- `auto:p60` = keep candidates above the 60th percentile density\n"
            "- `0.12` = keep candidates where density ≥ 0.12"
        ),
    )
    density_sigma = st.number_input(
        "Density smoothing scale (pixels)",
        min_value=1.0,
        max_value=500.0,
        key="cfg_density_sigma",
        step=1.0,
        help="Smoothing radius for density. Bigger = settlement-scale patterns; smaller = more local sensitivity.",
    )
    max_slope_deg = st.number_input(
        "Maximum region slope q75 (degrees)",
        min_value=0.0,
        max_value=89.0,
        key="cfg_max_slope_deg",
        step=0.5,
        help=(
            "Candidate regions are rejected if their 75th-percentile slope exceeds this value. "
            "Lower values suppress steep-slope artifacts."
        ),
    )
    consensus_enabled = st.checkbox(
        "Enable multi-threshold consensus",
        key="cfg_consensus_enabled",
        help=(
            "Runs candidate extraction across multiple positive-relief thresholds and keeps regions "
            "supported by repeated detections."
        ),
    )
    consensus_percentiles = st.text_input(
        "Consensus relief percentiles (comma-separated)",
        key="cfg_consensus_percentiles",
        disabled=not consensus_enabled,
        help="Example: `95,96,97`.",
    )
    consensus_min_support = st.number_input(
        "Consensus minimum support",
        min_value=1,
        max_value=10,
        key="cfg_consensus_min_support",
        step=1,
        disabled=not consensus_enabled,
        help="Minimum number of threshold runs that must support a candidate region.",
    )
    consensus_radius_m = st.number_input(
        "Consensus match radius (m)",
        min_value=0.0,
        max_value=100.0,
        key="cfg_consensus_radius_m",
        step=1.0,
        disabled=not consensus_enabled,
        help="Center-distance radius used to match detections across threshold runs.",
    )

    st.divider()
    st.markdown("### 5) Shape cleanup (remove noise-like shapes)")
    min_peak = st.number_input(
        "Minimum peak relief (m)",
        min_value=0.0,
        max_value=5.0,
        key="cfg_min_peak",
        step=0.05,
        help="Drops candidates whose strongest relief peak is below this (often tiny terrain wiggles).",
    )
    min_area_m2 = st.number_input(
        "Minimum footprint area (m²)",
        min_value=0.0,
        max_value=5000.0,
        key="cfg_min_area_m2",
        step=1.0,
        help="Drops very small candidate patches (area in square meters).",
    )
    max_area_m2 = st.number_input(
        "Maximum footprint area (m²)",
        min_value=0.0,
        max_value=50000.0,
        key="cfg_max_area_m2",
        step=10.0,
        help=(
            "Drops very large regions that are often broad terrain trends, ridges, or merged artifacts. "
            "Set to 0 to disable."
        ),
    )
    min_extent = st.number_input(
        "Minimum extent (0–1)",
        min_value=0.0,
        max_value=1.0,
        key="cfg_min_extent",
        step=0.01,
        help="Extent = area / bounding-box-area. Higher = more coherent; lower often = thin/noisy ridges.",
    )
    max_aspect = st.number_input(
        "Maximum elongation (≥1)",
        min_value=1.0,
        max_value=50.0,
        key="cfg_max_aspect",
        step=0.1,
        help="Elongation = max(width/height, height/width). High values are long/skinny shapes (often ridges/edges/artifacts).",
    )
    edge_buffer_m = st.number_input(
        "Tile-edge exclusion buffer (m)",
        min_value=0.0,
        max_value=200.0,
        key="cfg_edge_buffer_m",
        step=1.0,
        help=(
            "Drops regions near tile boundaries where partial cutoffs create artifacts. "
            "Set to 0 to disable."
        ),
    )
    min_spacing_m = st.number_input(
        "Minimum candidate spacing (m)",
        min_value=0.0,
        max_value=500.0,
        key="cfg_min_spacing_m",
        step=1.0,
        help=(
            "Score-ordered de-dup radius. Keeps the highest score within each local neighborhood. "
            "Set to 0 to disable."
        ),
    )
    min_prominence = st.number_input(
        "Minimum local prominence (m)",
        min_value=0.0,
        max_value=5.0,
        key="cfg_min_prominence",
        step=0.01,
        help=(
            "Prominence compares region mean relief to a surrounding ring. "
            "Higher values suppress broad background trends and ridge-like artifacts."
        ),
    )
    min_compactness = st.number_input(
        "Minimum compactness (0–1)",
        min_value=0.0,
        max_value=1.0,
        key="cfg_min_compactness",
        step=0.01,
        help=(
            "Compactness = 4*pi*Area / Perimeter^2. "
            "Lower values are line-like features (often drainage/ridges)."
        ),
    )
    min_solidity = st.number_input(
        "Minimum solidity (0–1)",
        min_value=0.0,
        max_value=1.0,
        key="cfg_min_solidity",
        step=0.01,
        help=(
            "Solidity = Area / Convex Hull Area. "
            "Lower values are fragmented or highly irregular features."
        ),
    )
    max_peak_offset = st.number_input(
        "Maximum peak offset ratio (0 = disabled)",
        min_value=0.0,
        max_value=3.0,
        value=st.session_state.get("cfg_max_peak_offset", 0.60),
        step=0.05,
        key="cfg_max_peak_offset",
        help=(
            "Peak offset ratio = distance(centroid → LRM peak pixel) / sqrt(area_pixels). "
            "Low values (< 0.5) characterize dome-shaped mounds with a central high point. "
            "High values suggest ridge segments, tree-throw mounds, or edge artifacts. "
            "Set to 0 to disable. Suggested starting point: 0.65."
        ),
    )
    max_tpi = st.number_input(
        "Maximum TPI at candidate (0 = disabled)",
        min_value=0.0,
        max_value=20.0,
        value=st.session_state.get("cfg_max_tpi", 2.0),
        step=0.25,
        key="cfg_max_tpi",
        help=(
            "Topographic Position Index (TPI) = elevation at the candidate centroid minus the "
            "mean elevation within the TPI radius. Ridge crests and hilltops score > 2 m; "
            "candidates on flat or gently rolling terrain score < 0.5 m. "
            "Enable (e.g. 2.0) to suppress false positives from natural ridges. "
            "Set to 0 to disable — recommended for hilly terrain where real sites occupy hilltops."
        ),
    )
    tpi_radius_m = st.number_input(
        "TPI radius (m)",
        min_value=10.0,
        max_value=500.0,
        value=st.session_state.get("cfg_tpi_radius_m", 50.0),
        step=10.0,
        key="cfg_tpi_radius_m",
        help="Neighbourhood radius used to compute TPI. Larger values capture broader topographic context.",
    )

    st.divider()
    st.markdown("### 6) Clustering (settlement patterns)")
    cluster_eps = st.text_input("Cluster radius (m)", key="cfg_cluster_eps", help="DBSCAN radius. Use `auto` or a number like `150`.")
    min_samples = st.number_input("Min candidates per cluster", min_value=1, max_value=50, key="cfg_min_samples", step=1)

    st.divider()
    st.markdown("### 7) Outputs")
    report_top_n = st.number_input("Featured candidates (Top N)", min_value=1, max_value=500, key="cfg_report_top_n", step=1)
    label_top_n = st.number_input("KML labeled points (Top N)", min_value=0, max_value=5000, key="cfg_label_top_n", step=5)

    st.divider()
    st.markdown("### 8) Depression detection (optional)")
    st.caption(
        "Run a second pass on the inverted LRM to detect negative-relief features: aguadas (reservoirs), "
        "plazas, quarries, and borrow pits. Depression candidates appear with a dashed blue marker (▼) in the report."
    )
    detect_depressions = st.checkbox(
        "Detect depressions",
        value=st.session_state.get("cfg_detect_depressions", False),
        key="cfg_detect_depressions",
        help="Inverts the LRM and runs the detection pipeline again to find bowl-shaped negative-relief features.",
    )

    st.divider()
    st.markdown("### 9) Validation (optional)")
    st.caption(
        "Provide a CSV (columns: name,lat,lon) or GeoJSON (Point features) with known archaeological "
        "site locations to measure recall. The HTML report will show a validation KPI bar and per-site markers."
    )
    validate_against = st.text_input(
        "Known sites file (path or leave blank)",
        value=st.session_state.get("cfg_validate_against", ""),
        key="cfg_validate_against",
        placeholder="data/caracol_structures_osm.csv",
        help=(
            "Path to a CSV with columns name,lat,lon or a GeoJSON FeatureCollection of Point features. "
            "Relative paths are resolved from the MayaScan project root."
        ),
    )
    validate_match_radius_m = st.number_input(
        "Match radius (m)",
        min_value=1.0, max_value=500.0,
        value=st.session_state.get("cfg_validate_match_radius_m", 25.0),
        step=5.0, key="cfg_validate_match_radius_m",
        help=(
            "A candidate counts as a hit if its centroid is within this radius of a known site. "
            "25 m is appropriate for structure-level matching at 1 m resolution."
        ),
    )

    st.divider()
    input_ready = (uploaded is not None) if mode == "Upload .laz/.las" else bool(str(input_local).strip())
    if not input_ready:
        if mode == "Upload .laz/.las":
            st.caption("Upload a `.laz/.las` file to enable run actions.")
        else:
            st.caption("Enter a local input path to enable run actions.")

    run_btn = st.button(
        "▶ Run MayaScan",
        type="primary",
        key="run_mayascan_btn",
        disabled=st.session_state.is_running or not input_ready,
    )
    st.markdown("### 8) Compare presets")
    compare_presets = st.multiselect(
        "Presets to compare",
        options=list(PRESET_VALUES.keys()),
        key="cfg_compare_presets",
        help="Runs each selected preset on the same input tile and summarizes differences.",
    )
    compare_presets = normalize_compare_presets(compare_presets)
    compare_count = len(compare_presets)
    st.caption(f"{compare_count} preset{'s' if compare_count != 1 else ''} selected.")
    if compare_count < 2:
        st.caption("Select at least two presets to enable comparison.")
    compare_btn = st.button(
        "▶ Run preset comparison",
        key="run_preset_compare_btn",
        disabled=st.session_state.is_running or compare_count < 2 or not input_ready,
        help="Select at least two presets to enable comparison.",
    )
    if st.session_state.is_running:
        st.caption("Run in progress... open `Run details` for live progress.")


# -----------------------------
# Main area: Tabs
# -----------------------------
tab_results, tab_compare, tab_runlogs, tab_glossary = st.tabs(["📍 Results", "📊 Comparison", "⚙️ Run details", "📖 Glossary"])

def resolve_input_and_run():
    st.session_state.last_compare_summary = None
    runs_dir_path = resolve_runs_dir(runs_dir)
    runs_dir_path.mkdir(parents=True, exist_ok=True)

    safe_run_name = sanitize_run_name(run_name)
    if safe_run_name != run_name.strip():
        st.warning(f"Run name sanitized to `{safe_run_name}` for filesystem safety.")

    pos_thresh_spec = pos_thresh.strip()
    min_density_spec = min_density.strip()
    cluster_eps_spec = cluster_eps.strip()
    consensus_percentiles_spec = str(consensus_percentiles).strip()

    errors: list[str] = []
    ok, err = validate_auto_or_numeric(
        pos_thresh_spec,
        field_name="Relief threshold",
        min_value=0.0,
    )
    if not ok and err:
        errors.append(err)

    ok, err = validate_auto_or_numeric(
        min_density_spec,
        field_name="Neighborhood density threshold",
        min_value=0.0,
        max_value=1.0,
    )
    if not ok and err:
        errors.append(err)

    ok, err = validate_cluster_eps(cluster_eps_spec)
    if not ok and err:
        errors.append(err)

    consensus_vals: list[float] | None = None
    if consensus_enabled:
        consensus_vals, err = parse_consensus_percentiles_csv(consensus_percentiles_spec)
        if err:
            errors.append(err)
        if int(consensus_min_support) < 1:
            errors.append("Consensus minimum support must be >= 1.")
        if float(consensus_radius_m) < 0.0:
            errors.append("Consensus match radius (m) must be >= 0.")
    if float(max_area_m2) > 0.0 and float(max_area_m2) < float(min_area_m2):
        errors.append("Maximum footprint area (m²) must be >= minimum footprint area, or 0 to disable.")

    if errors:
        for e in errors:
            st.error(e)
        render_hint_block(
            "Parameter validation failed.",
            [
                "Use `auto:pNN` (example: `auto:p96`) or a numeric value.",
                "For density, numeric values must stay between 0 and 1.",
                "For cluster radius, use `auto` or a positive number in meters.",
                "For consensus percentiles, use comma-separated values in [0,100] (example: `95,96,97`).",
            ],
            level="warning",
        )
        return None, None, None

    if cluster_eps_spec.lower() == "auto":
        cluster_eps_spec = "auto"
    if consensus_enabled and consensus_vals is not None:
        consensus_percentiles_spec = ",".join(f"{v:g}" for v in consensus_vals)

    input_path = resolve_input_path(mode, uploaded, input_local)
    if input_path is None:
        return None, None, None

    run_dir = runs_dir_path / safe_run_name
    if run_dir.exists() and not overwrite:
        auto_name = next_unique_run_name(runs_dir_path, safe_run_name)
        st.warning(
            "Run folder already exists for this name. "
            f"Auto-switching to `{auto_name}` so you can run again."
        )
        safe_run_name = auto_name
        run_dir = runs_dir_path / safe_run_name
        st.session_state.pending_cfg_run_name = safe_run_name

    cmd = build_cmd(
        input_path=input_path,
        run_name=safe_run_name,
        runs_dir=runs_dir_path,
        overwrite=overwrite,
        try_smrf=try_smrf,
        pos_thresh=pos_thresh_spec,
        min_density=min_density_spec,
        density_sigma=float(density_sigma),
        max_slope_deg=float(max_slope_deg),
        consensus_enabled=bool(consensus_enabled),
        consensus_percentiles=consensus_percentiles_spec,
        consensus_min_support=int(consensus_min_support),
        consensus_radius_m=float(consensus_radius_m),
        min_peak=float(min_peak),
        min_area_m2=float(min_area_m2),
        max_area_m2=float(max_area_m2),
        min_extent=float(min_extent),
        max_aspect=float(max_aspect),
        edge_buffer_m=float(edge_buffer_m),
        min_spacing_m=float(min_spacing_m),
        min_prominence=float(min_prominence),
        min_compactness=float(min_compactness),
        min_solidity=float(min_solidity),
        cluster_eps=cluster_eps_spec,
        min_samples=int(min_samples),
        report_top_n=int(report_top_n),
        label_top_n=int(label_top_n),
        no_html=no_html,
        resolution_m=float(resolution_m),
        smrf_scalar=float(smrf_scalar),
        smrf_slope=float(smrf_slope),
        smrf_threshold=float(smrf_threshold),
        smrf_window=float(smrf_window),
        max_peak_offset=float(max_peak_offset),
        tpi_radius_m=float(tpi_radius_m),
        max_tpi=float(max_tpi),
        detect_depressions=bool(detect_depressions),
        validate_against=str(validate_against),
        validate_match_radius_m=float(validate_match_radius_m),
    )

    st.session_state.last_run_dir = str(run_dir)
    st.session_state.last_cmd = cmd
    st.session_state.last_preset_selected = preset_name
    st.session_state.last_preset_match = bool(preset_match)
    st.session_state.last_ui_config = current_ui_config_snapshot()
    st.session_state.last_compare_summary = None
    st.session_state.zip_ready_for_run = None
    st.session_state.zip_path = None

    log_lines = []
    with tab_runlogs:
        st.markdown("### Run")
        if portfolio_mode:
            st.caption("Portfolio mode is on. Live diagnostics are hidden for a cleaner presentation.")
        else:
            st.caption("When finished, switch to **Results** to review the map, candidates, and report.")
            with st.expander("Command", expanded=False):
                st.code(" ".join(cmd), language="bash")

        status = st.empty()
        status.info("🏛️ Scanning terrain…")
        progress_bar = st.progress(0.02)
        log_box = None
        if not portfolio_mode:
            with st.expander("Live logs", expanded=True):
                log_box = st.empty()

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                log_lines.append(line.rstrip("\n"))
                if log_box is not None:
                    tail = "\n".join(log_lines[-450:])
                    log_box.code(tail, language="text")
                step_info = parse_step_from_log_line(line.rstrip("\n"))
                if step_info:
                    step_idx, step_title = step_info
                    # maya_scan has step 0..7
                    frac = max(0.05, min(0.95, float(step_idx + 1) / 8.0))
                    progress_bar.progress(frac)
                    status.info(f"🏛️ Step {step_idx}: {step_title}")
            if line == "" and proc.poll() is not None:
                break
            time.sleep(0.01)

        rc = proc.returncode

    st.session_state.last_logs = "\n".join(log_lines)

    with tab_runlogs:
        if rc == 0:
            progress_bar.progress(1.0)
            status.success(f"Done ✅  Output folder: {run_dir}")
            st.session_state.pending_cfg_run_name = next_default_run_name()
        else:
            status.error(f"Failed ❌  Exit code: {rc}")
            render_hint_block("Run failed. Try these fixes:", run_failure_hints(st.session_state.last_logs), level="error")

    return run_dir, runs_dir_path, cmd


def resolve_and_run_preset_comparison(selected_compare_presets: list[str]):
    st.session_state.last_compare_summary = None
    runs_dir_path = resolve_runs_dir(runs_dir)
    runs_dir_path.mkdir(parents=True, exist_ok=True)

    selected_compare_presets = normalize_compare_presets(selected_compare_presets)
    if len(selected_compare_presets) < 2:
        fallback = normalize_compare_presets([PRESET_BALANCED, PRESET_STRICT])
        if len(fallback) >= 2:
            selected_compare_presets = fallback
            st.warning(
                "Preset selection was incomplete at run-time. "
                "Defaulting comparison to Balanced + Strict."
            )

    safe_base_name = sanitize_run_name(run_name)
    if len(selected_compare_presets) < 2:
        st.error("Select at least two presets for comparison.")
        return None

    input_path = resolve_input_path(mode, uploaded, input_local)
    if input_path is None:
        return None

    compare_stamp = unique_stamp()
    run_specs: list[dict[str, object]] = []
    for p_name in selected_compare_presets:
        vals = PRESET_VALUES[p_name]
        suffix = preset_slug(p_name)
        run_name_i = sanitize_run_name(f"{safe_base_name}_{suffix}_{compare_stamp}")
        run_dir_i = runs_dir_path / run_name_i

        cmd_i = build_cmd(
            input_path=input_path,
            run_name=run_name_i,
            runs_dir=runs_dir_path,
            overwrite=overwrite,
            try_smrf=try_smrf,
            pos_thresh=str(vals["cfg_pos_thresh"]),
            min_density=str(vals["cfg_min_density"]),
            density_sigma=float(vals["cfg_density_sigma"]),
            max_slope_deg=float(vals["cfg_max_slope_deg"]),
            consensus_enabled=bool(vals["cfg_consensus_enabled"]),
            consensus_percentiles=str(vals["cfg_consensus_percentiles"]),
            consensus_min_support=int(vals["cfg_consensus_min_support"]),
            consensus_radius_m=float(vals["cfg_consensus_radius_m"]),
            min_peak=float(vals["cfg_min_peak"]),
            min_area_m2=float(vals["cfg_min_area_m2"]),
            max_area_m2=float(vals["cfg_max_area_m2"]),
            min_extent=float(vals["cfg_min_extent"]),
            max_aspect=float(vals["cfg_max_aspect"]),
            edge_buffer_m=float(vals["cfg_edge_buffer_m"]),
            min_spacing_m=float(vals["cfg_min_spacing_m"]),
            min_prominence=float(vals["cfg_min_prominence"]),
            min_compactness=float(vals["cfg_min_compactness"]),
            min_solidity=float(vals["cfg_min_solidity"]),
            max_peak_offset=float(vals.get("cfg_max_peak_offset", 0.60)),
            tpi_radius_m=float(vals.get("cfg_tpi_radius_m", 50.0)),
            max_tpi=float(vals.get("cfg_max_tpi", 0.0)),
            cluster_eps=str(vals["cfg_cluster_eps"]),
            min_samples=int(vals["cfg_min_samples"]),
            report_top_n=int(vals["cfg_report_top_n"]),
            label_top_n=int(vals["cfg_label_top_n"]),
            no_html=no_html,
            resolution_m=float(resolution_m),
            smrf_scalar=float(smrf_scalar),
            smrf_slope=float(smrf_slope),
            smrf_threshold=float(smrf_threshold),
            smrf_window=float(smrf_window),
        )
        run_specs.append(
            {
                "preset": p_name,
                "run_name": run_name_i,
                "run_dir": run_dir_i,
                "cmd": cmd_i,
            }
        )

    log_lines: list[str] = []
    run_summaries: list[dict[str, object]] = []
    n = len(run_specs)

    with tab_runlogs:
        st.markdown("### Preset comparison")
        if portfolio_mode:
            st.caption("Portfolio mode is on. Live diagnostics are hidden for a cleaner presentation.")
        else:
            st.caption("Running selected presets on the same input tile.")
            with st.expander("Commands", expanded=False):
                for spec in run_specs:
                    st.code(" ".join(spec["cmd"]), language="bash")

        status = st.empty()
        progress_bar = st.progress(0.02)
        log_box = None
        if not portfolio_mode:
            with st.expander("Live logs", expanded=True):
                log_box = st.empty()

        for i, spec in enumerate(run_specs):
            preset_i = str(spec["preset"])
            cmd_i = spec["cmd"]
            run_dir_i = Path(spec["run_dir"])
            status.info(f"🏛️ Running {preset_i} ({i + 1}/{n})...")

            proc = subprocess.Popen(
                cmd_i,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if line:
                    msg = line.rstrip("\n")
                    log_lines.append(f"[{preset_i}] {msg}")
                    if log_box is not None:
                        tail = "\n".join(log_lines[-450:])
                        log_box.code(tail, language="text")
                    step_info = parse_step_from_log_line(msg)
                    if step_info:
                        step_idx, step_title = step_info
                        local_frac = max(0.02, min(0.98, float(step_idx + 1) / 8.0))
                        global_frac = ((i + local_frac) / n)
                        progress_bar.progress(max(0.02, min(0.99, global_frac)))
                        status.info(f"🏛️ {preset_i}: Step {step_idx} - {step_title}")
                if line == "" and proc.poll() is not None:
                    break
                time.sleep(0.01)

            rc = proc.returncode
            if rc != 0:
                st.session_state.last_logs = "\n".join(log_lines)
                st.session_state.last_cmd = cmd_i
                st.session_state.last_compare_summary = None
                status.error(f"Failed ❌  {preset_i} exited with code {rc}.")
                render_hint_block("Preset comparison failed. Try these fixes:", run_failure_hints(st.session_state.last_logs), level="error")
                return None

            s = read_run_summary(run_dir_i)
            s["preset"] = preset_i
            s["run_name"] = str(spec["run_name"])
            run_summaries.append(s)
            progress_bar.progress(max(0.02, min(0.99, float(i + 1) / n)))

        progress_bar.progress(1.0)
        status.success("Preset comparison complete ✅")

    baseline = run_summaries[0] if run_summaries else {}
    base_candidates = baseline.get("candidates")
    base_top_score = baseline.get("top_score")
    for s in run_summaries:
        cand = s.get("candidates")
        top = s.get("top_score")
        try:
            s["d_candidates_vs_baseline"] = int(cand) - int(base_candidates) if cand is not None and base_candidates is not None else None
        except Exception:
            s["d_candidates_vs_baseline"] = None
        try:
            s["d_top_score_vs_baseline"] = round(float(top) - float(base_top_score), 4) if top is not None and base_top_score is not None else None
        except Exception:
            s["d_top_score_vs_baseline"] = None

    compare_payload: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path),
        "base_name": safe_base_name,
        "baseline_preset": baseline.get("preset"),
        "presets": [str(x) for x in selected_compare_presets],
        "runs": run_summaries,
    }
    json_path, md_path = write_preset_compare_artifacts(runs_dir_path, safe_base_name, compare_payload)
    compare_payload["comparison_json"] = str(json_path)
    compare_payload["comparison_md"] = str(md_path)

    focus_preset = (
        PRESET_BALANCED
        if PRESET_BALANCED in selected_compare_presets
        else str(selected_compare_presets[0])
    )
    focus_spec = next((s for s in run_specs if str(s["preset"]) == focus_preset), run_specs[0])
    st.session_state.last_run_dir = str(focus_spec["run_dir"])
    st.session_state.last_cmd = focus_spec["cmd"]
    st.session_state.last_logs = "\n".join(log_lines)
    st.session_state.last_preset_selected = focus_preset
    st.session_state.last_preset_match = True
    st.session_state.last_ui_config = dict(PRESET_VALUES.get(focus_preset, {}))
    st.session_state.last_compare_summary = compare_payload
    st.session_state.zip_ready_for_run = None
    st.session_state.zip_path = None
    st.session_state.pending_cfg_run_name = next_default_run_name()
    return compare_payload


if run_btn and not st.session_state.is_running and input_ready:
    st.session_state.is_running = True
    try:
        resolve_input_and_run()
    finally:
        st.session_state.is_running = False

if compare_btn and not st.session_state.is_running and input_ready:
    st.session_state.is_running = True
    try:
        resolve_and_run_preset_comparison(compare_presets)
    finally:
        st.session_state.is_running = False


# -----------------------------
# Results tab
# -----------------------------
with tab_results:
    st.markdown("### Review")

    compare_summary = st.session_state.get("last_compare_summary")
    if compare_summary:
        st.caption("Preset comparison results are available in the **Comparison** tab.")

    if not st.session_state.last_run_dir:
        render_hint_block(
            "No run loaded yet.",
            [
                "Choose a LAZ/LAS source in the sidebar.",
                "Pick a scientific preset (Balanced is recommended to start).",
                "Click `Run MayaScan` and return here for map, rankings, and exports.",
            ],
        )
    else:
        run_dir = Path(st.session_state.last_run_dir)
        if not run_dir.exists():
            render_hint_block(
                f"Run folder not found: {run_dir}",
                [
                    "Load an existing run from the sidebar dropdown.",
                    "Or run a new job with a fresh run name.",
                    "If this happened after cleanup, regenerate outputs with `Run MayaScan`.",
                ],
                level="warning",
            )
        else:
            wait_for_file(run_dir / "candidates.csv", timeout_s=6.0)
            wait_for_file(run_dir / "process.log", timeout_s=6.0)

            process_log = read_text_safely(run_dir / "process.log")
            used = parse_values_used(process_log)

            run_params_path = run_dir / "run_params.json"
            run_params_data: dict | None = None
            if run_params_path.exists():
                try:
                    parsed = json.loads(run_params_path.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        run_params_data = parsed
                except Exception:
                    run_params_data = None

            df_loaded = load_candidates(run_dir)
            if df_loaded is not None and annotation_mode:
                labels_df = load_candidate_labels(run_dir)
                df = merge_labels_into_candidates(df_loaded, labels_df)
            else:
                df = df_loaded
            quality = assess_run_quality(used, df)
            _, _, score_formula_numeric, _ = score_formula_strings(run_params_data)
            score_formula_readable_str, score_weights_df = score_formula_readable(run_params_data)

            quality_check_help: dict[str, str] = {
                "Candidate count in useful review range": "Enough candidates for review without overwhelming volume.",
                "At least one non-noise cluster": "Verifies DBSCAN found structure beyond all-noise outputs.",
                "Top score threshold": "Checks whether the strongest candidate has meaningful composite signal.",
                "Median score threshold": "Checks whether overall candidate quality is not driven by one outlier.",
                "Noise fraction threshold": "Checks that DBSCAN noise does not dominate kept candidates.",
            }

            detail_rows: list[dict[str, str]] = []
            for item in quality.get("check_details", []):
                if not isinstance(item, dict):
                    continue
                passed = item.get("passed")
                if passed is True:
                    status = "PASS"
                elif passed is False:
                    status = "FAIL"
                else:
                    status = "N/A"
                check_label = str(item.get("label", "Check"))
                detail_rows.append(
                    {
                        "Status": status,
                        "Check": check_label,
                        "Observed": str(item.get("observed", "—")),
                        "Target": str(item.get("target", "—")),
                        "Tooltip": quality_check_help.get(check_label, "Quality heuristic check."),
                    }
                )
            quality_summary = f"Triage quality: {quality['label']} ({quality['passed']}/{quality['total']} parameter checks)"
            with st.expander(quality_summary, expanded=False):
                st.warning(
                    "These checks measure whether parameter settings produced a plausible output range. "
                    "They do NOT indicate that real archaeological features were found. "
                    "Field verification is required before any candidate can be treated as a confirmed site.",
                    icon="⚠️",
                )
                if detail_rows:
                    detail_df = pd.DataFrame(detail_rows)
                    st.dataframe(
                        detail_df[["Status", "Check", "Observed", "Target", "Tooltip"]].rename(columns={"Tooltip": "Why it matters"}),
                        width="stretch",
                        hide_index=True,
                    )
                st.caption("Scores are relative within this run only and have no absolute meaning across runs.")
            with st.expander("Scoring formula explanation", expanded=False):
                st.code(score_formula_readable_str, language="text")
                st.caption(
                    "Higher exponent means that component influences ranking more strongly. "
                    "This score ranks candidates within a run; it is not a probability."
                )
                st.dataframe(score_weights_df, width="stretch", hide_index=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Candidates", used.get("candidates_kept", "—"))
            m2.metric("Relief threshold (m)", used.get("pos_thresh_m", "—"))
            m3.metric("Density threshold", used.get("min_density", "—"))
            m4.metric("Cluster radius (m)", used.get("dbscan_eps_m", "—"))
            m5.metric("Clusters", used.get("clusters_found", "—"))

            if used:
                st.caption(
                    f"Actual values (from process.log): "
                    f"relief={used.get('pos_thresh_m','—')} ({used.get('pos_thresh_spec','—')}), "
                    f"density={used.get('min_density','—')} ({used.get('min_density_spec','—')}), "
                    f"cluster_eps={used.get('dbscan_eps_m','—')} m."
                )
            if str(quality.get("label", "")).lower().startswith("weak"):
                next_steps = quality_action_hints(quality, used)
                if next_steps:
                    render_hint_block("Quick tuning steps for this weak/noisy run", next_steps, level="warning")

            filtered_df = df
            if df is not None and not df.empty:
                st.markdown("#### Review filters")
                if annotation_mode:
                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                else:
                    filter_col1, filter_col2 = st.columns(2)
                    filter_col3 = None

                min_score_filter: float | None = None
                with filter_col1:
                    if "score" in df.columns:
                        score_series = pd.to_numeric(df["score"], errors="coerce").dropna()
                        if not score_series.empty:
                            score_min = float(score_series.min())
                            score_max = float(score_series.max())
                            if score_max > score_min:
                                step = max((score_max - score_min) / 200.0, 0.001)
                                min_score_filter = st.slider(
                                    "Minimum score",
                                    min_value=score_min,
                                    max_value=score_max,
                                    value=score_min,
                                    step=step,
                                    format="%.3f",
                                    help=(
                                        "Composite ranking score filter. "
                                        "Higher score means higher review priority in this run. "
                                        "See 'Scoring formula explanation' above."
                                    ),
                                )
                            else:
                                min_score_filter = score_min
                                st.caption(f"All scores are {score_min:.3f}.")
                    else:
                        st.caption("No score column available; score filter disabled.")

                cluster_label_to_id: dict[str, int] = {}
                selected_cluster_labels: list[str] = ["All clusters"]
                with filter_col2:
                    if "cluster_id" in df.columns:
                        cluster_ids = sorted(pd.to_numeric(df["cluster_id"], errors="coerce").dropna().astype(int).unique().tolist())
                        if len(cluster_ids) <= 1:
                            st.caption("Cluster filter hidden (single cluster/noise group in this run).")
                        else:
                            cluster_label_to_id = {
                                ("Noise (-1)" if cid == -1 else f"Cluster {cid}"): cid
                                for cid in cluster_ids
                            }
                            cluster_options = ["All clusters"] + list(cluster_label_to_id.keys())
                            selected_cluster_labels = st.multiselect(
                                "Clusters",
                                options=cluster_options,
                                default=["All clusters"],
                            )
                    else:
                        st.caption("No cluster_id column available; cluster filter disabled.")

                selected_labels: list[str] | None = None
                if annotation_mode and filter_col3 is not None:
                    with filter_col3:
                        if "analyst_label" in df.columns:
                            label_options = ["likely", "unlikely", "unknown"]
                            selected_labels = st.multiselect(
                                "Analyst labels",
                                options=label_options,
                                default=label_options,
                            )
                        else:
                            st.caption("No analyst labels yet.")

                min_prominence_filter: float | None = None
                min_compactness_filter: float | None = None
                min_solidity_filter: float | None = None
                shape_col1, shape_col2, shape_col3 = st.columns(3)
                with shape_col1:
                    if "prominence_m" in df.columns:
                        prom_series = pd.to_numeric(df["prominence_m"], errors="coerce").dropna()
                        if not prom_series.empty:
                            pmin = float(prom_series.min())
                            pmax = float(prom_series.max())
                            min_prominence_filter = st.slider(
                                "Minimum prominence (m)",
                                min_value=pmin,
                                max_value=pmax,
                                value=pmin,
                                step=max((pmax - pmin) / 200.0, 0.001),
                                format="%.3f",
                            )
                with shape_col2:
                    if "compactness" in df.columns:
                        min_compactness_filter = st.slider(
                            "Minimum compactness",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.01,
                        )
                with shape_col3:
                    if "solidity" in df.columns:
                        min_solidity_filter = st.slider(
                            "Minimum solidity",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.01,
                        )

                filtered_df = df.copy()
                if min_score_filter is not None and "score" in filtered_df.columns:
                    filtered_df = filtered_df[pd.to_numeric(filtered_df["score"], errors="coerce") >= float(min_score_filter)]

                if "cluster_id" in filtered_df.columns and "All clusters" not in selected_cluster_labels:
                    allowed_ids = [cluster_label_to_id[label] for label in selected_cluster_labels if label in cluster_label_to_id]
                    filtered_df = filtered_df[filtered_df["cluster_id"].isin(allowed_ids)]

                if annotation_mode and "analyst_label" in filtered_df.columns and selected_labels:
                    filtered_df = filtered_df[
                        filtered_df["analyst_label"].astype(str).str.lower().isin([s.lower() for s in selected_labels])
                    ]
                elif annotation_mode and "analyst_label" in filtered_df.columns and not selected_labels:
                    filtered_df = filtered_df.iloc[0:0]

                if min_prominence_filter is not None and "prominence_m" in filtered_df.columns:
                    filtered_df = filtered_df[pd.to_numeric(filtered_df["prominence_m"], errors="coerce") >= float(min_prominence_filter)]
                if min_compactness_filter is not None and "compactness" in filtered_df.columns:
                    filtered_df = filtered_df[pd.to_numeric(filtered_df["compactness"], errors="coerce") >= float(min_compactness_filter)]
                if min_solidity_filter is not None and "solidity" in filtered_df.columns:
                    filtered_df = filtered_df[pd.to_numeric(filtered_df["solidity"], errors="coerce") >= float(min_solidity_filter)]

                st.caption(f"Showing {len(filtered_df)} of {len(df)} candidates after filters.")

            report_html = run_dir / "report.html"
            report_md = run_dir / "report.md"
            run_params_json = run_dir / "run_params.json"
            candidates_csv = run_dir / "candidates.csv"
            labels_csv = labels_path_for_run(run_dir)
            geojson = run_dir / "candidates.geojson"
            kml = run_dir / "candidates.kml"
            plots_dir = run_dir / "plots"
            img_dir = run_dir / "html" / "img"

            acct = {}
            if isinstance(run_params_data, dict):
                acct_raw = run_params_data.get("candidate_accounting")
                if isinstance(acct_raw, dict):
                    acct = acct_raw
            dropped_edge = _to_int_or_none(acct.get("dropped_edge_buffer", used.get("dropped_edge")))
            dropped_consensus = _to_int_or_none(acct.get("dropped_consensus_support", used.get("dropped_consensus")))
            dropped_density = _to_int_or_none(acct.get("dropped_density", used.get("dropped_density")))
            dropped_post = _to_int_or_none(acct.get("dropped_post_filters", used.get("dropped_post")))
            dropped_spacing = _to_int_or_none(acct.get("dropped_spacing_dedup", used.get("dropped_spacing")))
            kept = _to_int_or_none(acct.get("kept_candidates", used.get("candidates_kept")))

            stage_metrics_raw = {}
            if isinstance(run_params_data, dict):
                sm = run_params_data.get("stage_metrics_sec")
                if isinstance(sm, dict):
                    stage_metrics_raw = sm

            res_tabs = st.tabs(["🗺️ Map", "🏷️ Top candidates", "📋 Table", "🧾 Report", "🧪 Diagnostics", "📄 Files", "📈 Plots"])

            # --- Map (Leaflet: always works, Street + Satellite)
            with res_tabs[0]:
                if filtered_df is None or filtered_df.empty:
                    if df is not None and not df.empty:
                        render_hint_block(
                            "No candidates match current filters.",
                            [
                                "Lower the minimum score slider.",
                                "Include more clusters (or choose All clusters).",
                                "If needed, rerun with less strict cleanup thresholds.",
                            ],
                        )
                    else:
                        render_hint_block(
                            "No candidates available for this run.",
                            [
                                "Open **Run details** to confirm the run completed successfully.",
                                "Try a less strict preset (`Balanced` or `Exploratory`).",
                                "Lower `min_peak`, `min_area_m2`, or `min_extent` slightly and rerun.",
                            ],
                            level="warning",
                        )
                else:
                    st.markdown("#### Candidates map")
                    components.html(leaflet_map_html(filtered_df), height=740, scrolling=False)
                    st.caption("Street and Satellite layers are available in the top-right map control. Click points for details.")

            # --- Top candidates + cutouts
            with res_tabs[1]:
                if filtered_df is None or filtered_df.empty:
                    if df is not None and not df.empty:
                        render_hint_block(
                            "No candidates match current filters.",
                            [
                                "Lower minimum score.",
                                "Reset cluster filter to All clusters.",
                            ],
                        )
                    else:
                        render_hint_block(
                            "No candidate table found yet.",
                            [
                                "Run MayaScan first, then return to Results.",
                                "If run already finished, check `runs/<run_name>/candidates.csv` exists.",
                            ],
                            level="warning",
                        )
                else:
                    topn = min(30, len(filtered_df))
                    top = filtered_df.sort_values("score", ascending=False).head(topn).copy()
                    top_base_cols = [
                        "cand_id",
                        "score",
                        "density",
                        "peak_relief_m",
                        "consensus_support",
                        "prominence_m",
                        "area_m2",
                        "extent",
                        "aspect",
                        "compactness",
                        "solidity",
                        "cluster_id",
                        "lat",
                        "lon",
                    ]
                    if annotation_mode:
                        top_base_cols.insert(12, "analyst_label")
                    top_cols = [
                        c
                        for c in top_base_cols
                        if c in top.columns
                    ]
                    top_display = top[top_cols].copy()
                    for fcol, ndigits in {
                        "score": 3,
                        "density": 3,
                        "peak_relief_m": 2,
                        "prominence_m": 2,
                        "area_m2": 1,
                        "extent": 3,
                        "aspect": 2,
                        "compactness": 3,
                        "solidity": 3,
                        "lat": 6,
                        "lon": 6,
                    }.items():
                        if fcol in top_display.columns:
                            top_display[fcol] = pd.to_numeric(top_display[fcol], errors="coerce").round(ndigits)
                    st.dataframe(top_display, width="stretch", hide_index=True)

                    if "cand_id" not in top.columns:
                        st.info("No cand_id column available for candidate inspection.")
                    else:
                        top_view = top.copy()
                        top_view["_cand_id_int"] = pd.to_numeric(top_view["cand_id"], errors="coerce").astype("Int64")
                        top_view = top_view.dropna(subset=["_cand_id_int"]).copy()
                        top_view["_cand_id_int"] = top_view["_cand_id_int"].astype(int)

                        if top_view.empty:
                            st.info("No valid cand_id values available for candidate inspection.")
                        else:
                            option_ids = top_view["_cand_id_int"].tolist()

                            def _label_for_cid(cid: int) -> str:
                                row = top_view[top_view["_cand_id_int"] == cid].iloc[0]
                                try:
                                    score_val = float(row.get("score"))
                                    return f"Candidate {cid} (score {score_val:.3f})"
                                except Exception:
                                    return f"Candidate {cid}"

                            selected_cid = st.selectbox(
                                "Inspect candidate",
                                options=option_ids,
                                format_func=_label_for_cid,
                                key=f"inspect_cand_{run_dir.name}",
                            )

                            selected_row = top_view[top_view["_cand_id_int"] == int(selected_cid)].iloc[0]
                            st.markdown("#### Candidate detail")
                            detail_tab_labels = ["Summary"]
                            if annotation_mode:
                                detail_tab_labels.append("Label")
                            detail_tab_labels.extend(["Explainability", "Panel"])
                            detail_tabs = st.tabs(detail_tab_labels)
                            explain_idx = 2 if annotation_mode else 1
                            panel_idx = 3 if annotation_mode else 2

                            with detail_tabs[0]:
                                s1, s2, s3, s4 = st.columns(4)
                                s1.metric(
                                    "Score",
                                    f"{float(selected_row['score']):.3f}" if "score" in selected_row else "—",
                                    help=(
                                        "Composite ranking metric used by MayaScan. "
                                        "Higher means higher review priority within this run. "
                                        "See 'Scoring formula explanation' near the top of Results."
                                    ),
                                )
                                s2.metric("Peak relief (m)", f"{float(selected_row['peak_relief_m']):.2f}" if "peak_relief_m" in selected_row else "—")
                                s3.metric("Support", f"{int(selected_row['consensus_support'])}" if "consensus_support" in selected_row and pd.notna(selected_row["consensus_support"]) else "—")
                                s4.metric("Prominence (m)", f"{float(selected_row['prominence_m']):.2f}" if "prominence_m" in selected_row else "—")
                                s5, s6, s7, s8 = st.columns(4)
                                s5.metric("Area (m²)", f"{float(selected_row['area_m2']):.1f}" if "area_m2" in selected_row else "—")
                                s6.metric("Extent", f"{float(selected_row['extent']):.3f}" if "extent" in selected_row else "—")
                                s7.metric("Compactness", f"{float(selected_row['compactness']):.3f}" if "compactness" in selected_row else "—")
                                s8.metric("Solidity", f"{float(selected_row['solidity']):.3f}" if "solidity" in selected_row else "—")
                                if "cluster_id" in selected_row and pd.notna(selected_row["cluster_id"]):
                                    st.caption(f"Cluster: {int(selected_row['cluster_id'])}")
                                if "lat" in selected_row and "lon" in selected_row and pd.notna(selected_row["lat"]) and pd.notna(selected_row["lon"]):
                                    lat = float(selected_row["lat"])
                                    lon = float(selected_row["lon"])
                                    st.markdown(f"[Open in Google Maps](https://www.google.com/maps?q={lat:.8f},{lon:.8f})")

                            if annotation_mode:
                                with detail_tabs[1]:
                                    cur_label = str(selected_row.get("analyst_label", "unknown")).strip().lower()
                                    if cur_label not in {"likely", "unlikely", "unknown"}:
                                        cur_label = "unknown"
                                    cur_note = str(selected_row.get("analyst_note", "") or "")
                                    cur_updated = str(selected_row.get("analyst_updated_utc", "") or "")
                                    label_options = ["unknown", "likely", "unlikely"]
                                    default_idx = label_options.index(cur_label) if cur_label in label_options else 0

                                    with st.form(key=f"label_form_{run_dir.name}_{int(selected_cid)}", clear_on_submit=False):
                                        new_label = st.selectbox("Classification", options=label_options, index=default_idx)
                                        new_note = st.text_input("Notes (optional)", value=cur_note, max_chars=500)
                                        save_label = st.form_submit_button("Save label")

                                    if cur_updated:
                                        st.caption(f"Last label update: {cur_updated}")

                                    if save_label:
                                        lat_v = _to_float_or_none(selected_row.get("lat"))
                                        lon_v = _to_float_or_none(selected_row.get("lon"))
                                        score_v = _to_float_or_none(selected_row.get("score"))
                                        labels_path = upsert_candidate_label(
                                            run_dir=run_dir,
                                            cand_id=int(selected_cid),
                                            label=new_label,
                                            note=new_note,
                                            lat=lat_v,
                                            lon=lon_v,
                                            score=score_v,
                                        )
                                        st.success(f"Saved label to {labels_path.name}.")
                                        st.rerun()

                            with detail_tabs[explain_idx]:
                                breakdown = score_breakdown_df(selected_row, run_params_data)
                                if breakdown is not None:
                                    st.caption(
                                        "Component-level contribution for this candidate "
                                        "(each term = component_value ^ exponent)."
                                    )
                                    st.dataframe(
                                        breakdown[["component", "raw_value", "exponent", "term"]],
                                        width="stretch",
                                        hide_index=True,
                                    )
                                    st.bar_chart(
                                        breakdown.set_index("component")[["term_norm"]],
                                        width="stretch",
                                    )
                                    est_score = float(breakdown.attrs.get("estimated_score", 0.0))
                                    obs_score = _to_float_or_none(selected_row.get("score"))
                                    if obs_score is not None:
                                        st.caption(
                                            f"Estimated score from components: {est_score:.6f} | "
                                            f"Observed score: {obs_score:.6f}."
                                        )
                                else:
                                    st.caption("Score breakdown is unavailable for this candidate row.")

                            with detail_tabs[panel_idx]:
                                if img_dir.exists():
                                    panel_path = img_dir / f"cand_{int(selected_cid):04d}_panel.png"
                                    if panel_path.exists():
                                        try:
                                            st.image(
                                                str(panel_path),
                                                caption=f"Candidate {int(selected_cid)} panel (LRM + hillshade)",
                                                width="stretch",
                                            )
                                        except Exception:
                                            st.warning(f"Candidate panel could not be displayed: `{panel_path.name}`")
                                    else:
                                        render_hint_block(
                                            "No cutout panel found for this candidate.",
                                            [
                                                "Only top-ranked candidates get cutouts.",
                                                "Increase `Featured candidates (Top N)` and rerun.",
                                                "Ensure HTML output is enabled (disable `Skip HTML report`).",
                                            ],
                                        )
                                else:
                                    render_hint_block(
                                        "No cutout image folder found for this run.",
                                        [
                                            "Enable HTML output and rerun to generate cutouts.",
                                        ],
                                        level="warning",
                                    )

            # --- Table
            with res_tabs[2]:
                if filtered_df is None or filtered_df.empty:
                    if df is not None and not df.empty:
                        st.info("No candidates match the current filters.")
                    else:
                        st.info("No candidates.csv found yet.")
                else:
                    table_df = filtered_df.sort_values("score", ascending=False).copy()
                    for fcol, ndigits in {
                        "score": 3,
                        "density": 3,
                        "peak_relief_m": 2,
                        "prominence_m": 2,
                        "area_m2": 1,
                        "extent": 3,
                        "aspect": 2,
                        "compactness": 3,
                        "solidity": 3,
                        "lat": 6,
                        "lon": 6,
                    }.items():
                        if fcol in table_df.columns:
                            table_df[fcol] = pd.to_numeric(table_df[fcol], errors="coerce").round(ndigits)
                    st.dataframe(table_df, width="stretch", hide_index=True)

            # --- Report
            with res_tabs[3]:
                if report_html.exists():
                    st.markdown("#### Report (interactive)")
                    rcol1, rcol2 = st.columns([1, 2])
                    with rcol1:
                        st.download_button(
                            "Download report.html",
                            data=report_html.read_bytes(),
                            file_name=report_html.name,
                            mime="text/html",
                            key=f"download_report_tab_{run_dir.name}",
                        )
                    with rcol2:
                        st.caption(f"Source file: `{report_html}`")

                    report_height = st.slider(
                        "Report viewport height (px)",
                        min_value=780,
                        max_value=1800,
                        value=1240,
                        step=40,
                        key=f"report_height_{run_dir.name}",
                    )
                    html_inlined = inline_report_images_and_basemap(
                        str(report_html),
                        str(run_dir),
                        report_html.stat().st_mtime_ns,
                    )
                    if html_inlined.strip():
                        components.html(html_inlined, height=int(report_height), scrolling=True)
                    else:
                        st.warning("report.html exists but could not be loaded as text.")
                else:
                    render_hint_block(
                        "No report.html found for this run.",
                        [
                            "You may have enabled `Skip HTML report`.",
                            "Rerun with HTML enabled to generate interactive report + cutouts.",
                        ],
                    )

            # --- Diagnostics
            with res_tabs[4]:
                st.markdown("#### Diagnostics")
                with st.expander("Quality rationale", expanded=False):
                    for label, ok in quality["checks"]:
                        st.write(f"{'✅' if ok else '⚠️'} {label}")
                    st.caption("This is a triage heuristic for review confidence, not archaeological validation.")

                if annotation_mode:
                    st.caption("Analyst labels are manual review notes and are not used as a primary run-quality metric.")

                if any(v is not None for v in [dropped_edge, dropped_consensus, dropped_density, dropped_post, dropped_spacing, kept]):
                    st.markdown("##### Filter waterfall")
                    wf_rows = [
                        {"stage": "Dropped edge", "count": int(dropped_edge or 0)},
                        {"stage": "Dropped consensus", "count": int(dropped_consensus or 0)},
                        {"stage": "Dropped density", "count": int(dropped_density or 0)},
                        {"stage": "Dropped post-filters", "count": int(dropped_post or 0)},
                        {"stage": "Dropped spacing dedup", "count": int(dropped_spacing or 0)},
                        {"stage": "Kept candidates", "count": int(kept or 0)},
                    ]
                    wf_df = pd.DataFrame(wf_rows)
                    st.bar_chart(wf_df.set_index("stage"), width="stretch")
                    st.caption("Use this to diagnose whether strictness is coming from edge/density/shape/spacing filters.")
                    tuning_hints = tuning_hints_from_filter_waterfall(
                        dropped_edge=dropped_edge,
                        dropped_consensus=dropped_consensus,
                        dropped_density=dropped_density,
                        dropped_post=dropped_post,
                        dropped_spacing=dropped_spacing,
                        kept=kept,
                    )
                    if tuning_hints:
                        render_hint_block("Tuning suggestions from this run", tuning_hints, level="info")

                if stage_metrics_raw:
                    stage_rows: list[dict[str, object]] = []
                    for k, v in stage_metrics_raw.items():
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        stage_rows.append({"stage": str(k), "seconds": fv})
                    if stage_rows:
                        stage_df = pd.DataFrame(stage_rows).sort_values("seconds", ascending=False)
                        st.markdown("##### Runtime profile")
                        st.bar_chart(stage_df.set_index("stage")[["seconds"]], width="stretch")
                        total_rt = next((r["seconds"] for r in stage_rows if str(r["stage"]) == "total_runtime_sec"), None)
                        if isinstance(total_rt, float):
                            st.caption(f"Total runtime: {total_rt:.2f} sec")

                if not portfolio_mode:
                    with st.expander("Run settings used", expanded=False):
                        settings_data: dict[str, object] = {"run_dir": str(run_dir)}
                        settings_data["ui_preset_selected"] = st.session_state.get("last_preset_selected")
                        settings_data["ui_preset_match"] = st.session_state.get("last_preset_match")
                        if st.session_state.last_cmd:
                            settings_data["command"] = parse_cmd_settings(st.session_state.last_cmd)
                        if st.session_state.get("last_ui_config"):
                            settings_data["ui_config_snapshot"] = st.session_state["last_ui_config"]
                        if run_params_data is not None:
                            settings_data["run_params_json"] = run_params_data
                        if used:
                            settings_data["resolved_from_log"] = {
                                "relief_threshold_m": used.get("pos_thresh_m"),
                                "relief_threshold_spec": used.get("pos_thresh_spec"),
                                "min_density": used.get("min_density"),
                                "min_density_spec": used.get("min_density_spec"),
                                "dbscan_eps_m": used.get("dbscan_eps_m"),
                                "dbscan_min_samples": used.get("dbscan_min_samples"),
                                "candidates_kept": used.get("candidates_kept"),
                                "clusters_found": used.get("clusters_found"),
                                "clusters_noise": used.get("clusters_noise"),
                                "dropped_edge": used.get("dropped_edge"),
                                "dropped_consensus": used.get("dropped_consensus"),
                                "dropped_density": used.get("dropped_density"),
                                "dropped_post": used.get("dropped_post"),
                                "dropped_spacing": used.get("dropped_spacing"),
                            }

                        report_md_path = run_dir / "report.md"
                        if report_md_path.exists():
                            m = re.search(r"- Input:\s*`([^`]+)`", read_text_safely(report_md_path))
                            if m:
                                settings_data["input"] = m.group(1)
                        st.json(settings_data)

                with st.expander("Provenance (copy / paste)", expanded=False):
                    prov_text = build_provenance_text(
                        run_dir=run_dir,
                        command=st.session_state.last_cmd,
                        used=used,
                        run_params_data=run_params_data,
                        preset_name=st.session_state.get("last_preset_selected"),
                        preset_match=st.session_state.get("last_preset_match"),
                    )
                    st.text_area("Provenance block", value=prov_text, height=220, key=f"prov_{run_dir.name}")
                    st.download_button(
                        "Download provenance.txt",
                        data=prov_text.encode("utf-8"),
                        file_name=f"{run_dir.name}_provenance.txt",
                        mime="text/plain",
                    )

            # --- Files
            with res_tabs[5]:
                st.markdown("#### Key outputs")
                st.code(str(run_dir), language="text")

                cols = st.columns(3)
                with cols[0]:
                    st.write("**Candidates table**")
                    if candidates_csv.exists():
                        st.success("candidates.csv")
                        st.download_button(
                            "Download candidates.csv",
                            data=candidates_csv.read_bytes(),
                            file_name=candidates_csv.name,
                            mime="text/csv",
                        )
                    else:
                        st.warning("candidates.csv not found")
                    if annotation_mode and labels_csv.exists():
                        st.success("candidate_labels.csv")
                        st.download_button(
                            "Download candidate_labels.csv",
                            data=labels_csv.read_bytes(),
                            file_name=labels_csv.name,
                            mime="text/csv",
                        )

                with cols[1]:
                    st.write("**GIS exports**")
                    if geojson.exists():
                        st.success("candidates.geojson")
                        st.download_button(
                            "Download candidates.geojson",
                            data=geojson.read_bytes(),
                            file_name=geojson.name,
                            mime="application/geo+json",
                        )
                    if kml.exists():
                        st.success("candidates.kml")
                        st.download_button(
                            "Download candidates.kml",
                            data=kml.read_bytes(),
                            file_name=kml.name,
                            mime="application/vnd.google-earth.kml+xml",
                        )

                with cols[2]:
                    st.write("**Reports**")
                    if report_html.exists():
                        st.success("report.html")
                        st.download_button(
                            "Download report.html",
                            data=report_html.read_bytes(),
                            file_name=report_html.name,
                            mime="text/html",
                        )
                    if report_md.exists():
                        st.success("report.md")
                        st.download_button(
                            "Download report.md",
                            data=report_md.read_bytes(),
                            file_name=report_md.name,
                            mime="text/markdown",
                        )
                    if run_params_json.exists():
                        st.success("run_params.json")
                        st.download_button(
                            "Download run_params.json",
                            data=run_params_json.read_bytes(),
                            file_name=run_params_json.name,
                            mime="application/json",
                        )

                st.divider()
                st.markdown("#### Download everything")
                zip_path = run_dir.parent / f"{run_dir.name}_outputs.zip"
                if st.button("Prepare run outputs (.zip)", key=f"prepare_zip_{run_dir.name}"):
                    with st.spinner("Preparing ZIP archive..."):
                        zip_run_dir(run_dir, zip_path)
                    st.session_state.zip_ready_for_run = str(run_dir)
                    st.session_state.zip_path = str(zip_path)

                zip_ready = (
                    st.session_state.get("zip_ready_for_run") == str(run_dir)
                    and st.session_state.get("zip_path")
                    and Path(st.session_state["zip_path"]).exists()
                )

                if zip_ready:
                    zip_ready_path = Path(st.session_state["zip_path"])
                    st.download_button(
                        label="Download run outputs (.zip)",
                        data=zip_ready_path.read_bytes(),
                        file_name=zip_ready_path.name,
                        mime="application/zip",
                    )
                else:
                    st.caption('Click "Prepare run outputs (.zip)" to generate the archive.')

            # --- Plots
            with res_tabs[6]:
                st.markdown("#### Plots")
                if plots_dir.exists():
                    time.sleep(0.15)
                    pngs = sorted(plots_dir.glob("*.png"))
                    if not pngs:
                        st.info("No plot PNGs found.")
                    else:
                        for p in pngs:
                            if not wait_for_file(p, timeout_s=2.0, poll_s=0.1):
                                st.caption(f"Skipping plot not ready yet: `{p.name}`")
                                continue
                            try:
                                st.image(str(p), caption=p.name, width="stretch")
                            except Exception:
                                st.warning(f"Skipped unreadable plot image: `{p.name}`")
                else:
                    st.info("plots/ folder not found.")


# -----------------------------
# Comparison tab
# -----------------------------
with tab_compare:
    st.markdown("### Preset comparison")
    compare_summary = st.session_state.get("last_compare_summary")
    if not compare_summary:
        render_hint_block(
            "No comparison has been run yet.",
            [
                "In the sidebar, choose at least two presets under `Compare presets`.",
                "Click `Run preset comparison`.",
                "Return here for side-by-side bars and deltas.",
            ],
        )
    else:
        baseline = compare_summary.get("baseline_preset")
        if baseline:
            st.caption(f"Baseline preset: {baseline}")

        rows = compare_summary.get("runs", [])
        cmp_df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
        if cmp_df.empty:
            st.warning("Comparison payload exists, but run rows are empty.")
        else:
            for col in ["candidates", "clusters", "noise", "top_score", "mean_score", "median_score", "d_candidates_vs_baseline", "d_top_score_vs_baseline"]:
                if col in cmp_df.columns:
                    cmp_df[col] = pd.to_numeric(cmp_df[col], errors="coerce")

            if "preset" in cmp_df.columns:
                cmp_df = cmp_df.sort_values("preset")

            st.markdown("#### Visual bars")
            bar_cols = [c for c in ["candidates", "clusters", "top_score", "mean_score"] if c in cmp_df.columns]
            if bar_cols and "preset" in cmp_df.columns:
                st.bar_chart(cmp_df.set_index("preset")[bar_cols], width="stretch")
            else:
                st.info("No plottable comparison metrics found.")

            if "preset" in cmp_df.columns:
                st.markdown("#### Delta vs baseline")
                for _, row in cmp_df.iterrows():
                    p_name = str(row.get("preset", "preset"))
                    if baseline and p_name == str(baseline):
                        continue
                    with st.expander(f"{p_name} vs baseline", expanded=False):
                        c1, c2, c3 = st.columns(3)
                        cand = row.get("candidates")
                        d_cand = row.get("d_candidates_vs_baseline")
                        top = row.get("top_score")
                        d_top = row.get("d_top_score_vs_baseline")
                        med = row.get("median_score")
                        c1.metric(
                            "Candidates",
                            "—" if pd.isna(cand) else f"{int(cand)}",
                            None if pd.isna(d_cand) else f"{int(d_cand):+d}",
                        )
                        c2.metric(
                            "Top score",
                            "—" if pd.isna(top) else f"{float(top):.3f}",
                            None if pd.isna(d_top) else f"{float(d_top):+.3f}",
                        )
                        c3.metric(
                            "Median score",
                            "—" if pd.isna(med) else f"{float(med):.3f}",
                        )

            st.markdown("#### Full comparison table")
            show_cols = [
                c for c in [
                    "preset",
                    "run_name",
                    "candidates",
                    "clusters",
                    "noise",
                    "top_score",
                    "mean_score",
                    "median_score",
                    "d_candidates_vs_baseline",
                    "d_top_score_vs_baseline",
                ]
                if c in cmp_df.columns
            ]
            if show_cols:
                st.dataframe(cmp_df[show_cols], width="stretch")

            if "run_dir" in cmp_df.columns and "preset" in cmp_df.columns:
                st.markdown("#### Open one compared run in Results")
                run_options: dict[str, str] = {}
                for _, row in cmp_df.iterrows():
                    preset_label = str(row.get("preset", "preset"))
                    run_name_label = str(row.get("run_name", "run"))
                    run_dir_label = str(row.get("run_dir", ""))
                    if not run_dir_label:
                        continue
                    run_options[f"{preset_label} — {run_name_label}"] = run_dir_label
                if run_options:
                    selected_run_label = st.selectbox(
                        "Comparison run",
                        options=list(run_options.keys()),
                        key="cmp_run_selector",
                    )
                    if st.button("Load selected comparison run", key="load_cmp_run_btn"):
                        st.session_state.last_run_dir = run_options[selected_run_label]
                        st.success(f"Loaded `{selected_run_label}` into Results.")
                        st.rerun()

        cjson = compare_summary.get("comparison_json")
        cmd = compare_summary.get("comparison_md")
        d1, d2 = st.columns(2)
        with d1:
            if cjson:
                p = Path(str(cjson))
                if p.exists():
                    st.download_button(
                        "Download comparison JSON",
                        data=p.read_bytes(),
                        file_name=p.name,
                        mime="application/json",
                    )
        with d2:
            if cmd:
                p = Path(str(cmd))
                if p.exists():
                    st.download_button(
                        "Download comparison markdown",
                        data=p.read_bytes(),
                        file_name=p.name,
                        mime="text/markdown",
                    )


# -----------------------------
# Run details tab (show last logs)
# -----------------------------
with tab_runlogs:
    if portfolio_mode:
        render_hint_block(
            "Portfolio mode is enabled.",
            [
                "Detailed command and logs are hidden in this mode.",
                "Disable `Portfolio mode` in the sidebar to inspect diagnostics.",
            ],
        )
    else:
        if st.session_state.get("last_compare_summary"):
            st.markdown("### Last preset comparison")
            st.json(st.session_state["last_compare_summary"])

        if st.session_state.last_cmd:
            st.markdown("### Last run")
            with st.expander("Command", expanded=False):
                st.code(" ".join(st.session_state.last_cmd), language="bash")
            with st.expander("Logs", expanded=False):
                st.code(
                    st.session_state.last_logs[-20000:] if st.session_state.last_logs else "(no logs yet)",
                    language="text",
                )
        else:
            st.info("Run MayaScan to see details here.")


# -----------------------------
# Glossary tab
# -----------------------------
with tab_glossary:
    st.markdown("### Glossary (plain-English)")
    st.markdown(
        """
**DTM (Digital Terrain Model)**  
A “bare earth” elevation raster derived from the point cloud.  
Units: meters above datum.

**LRM (Local Relief Model)**  
A terrain-enhancement layer that highlights subtle bumps/edges by subtracting a smoothed terrain from a less-smoothed terrain.

**Relief threshold**  
How strong a bump must be in the LRM to become a candidate region.  
`auto:p96` means “use the 96th percentile of positive relief values” for that tile.

**Candidate region**  
A connected patch of pixels above threshold (after cleanup and region-slope q75 filtering).

**Neighborhood density**  
A smoothed “how many candidates are nearby” signal. Helps emphasize settlement-like zones and suppress isolated noise.
Unitless (normalized 0-1).

**Extent (bbox fill, 0–1)**  
How filled-in a region is: area / bounding-box-area.  
Higher = more coherent; lower often = thin/noisy ridges.
Unitless.

**Compactness (0–1)**  
Defined as `4*pi*Area / Perimeter^2`.  
Lower values are more line-like and often correspond to drainage/ridge artifacts.
Unitless.

**Solidity (0–1)**  
Defined as `Area / ConvexHullArea`.  
Lower values indicate fragmented/irregular shapes; higher values are more solid footprints.
Unitless.

**Local prominence (m)**  
Region mean relief minus mean relief of a surrounding ring.  
Higher values indicate stand-out terrain features against local background.

**Edge buffer (m)**  
Excludes regions near raster boundaries to reduce tile-cut artifacts.

**Spacing de-dup (m)**  
Keeps the highest-scoring candidate within a local radius to reduce duplicate nearby points.

**Elongation (aspect ratio, ≥1)**  
How stretched a region is.  
High values are long/skinny shapes (often ridges/edges/artifacts).
Unitless.

**Score**  
Multiplicative ranking function using density/relief/shape terms.  
Higher score means stronger priority for review. It is **not** a calibrated probability of archaeology.
Formula (current): `density^a * peak^b * extent^c * prominence^d * compactness^e * solidity^f * area^g`.

**DBSCAN clustering**  
Groups candidates into clusters based on distance in meters (useful for settlement patterns).

**Scientific presets (Strict / Balanced / Exploratory)**  
Reproducible parameter profiles to trade precision vs recall before manual tuning.

**Preset comparison run**  
Runs multiple presets on the same input tile and reports side-by-side metric deltas.
"""
    )
