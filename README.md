# MayaScan

**LiDAR terrain-anomaly detection for archaeological review**

MayaScan is a Python geospatial pipeline for turning raw LAZ/LAS point clouds into ranked candidate features for analyst review. It builds terrain rasters, detects positive-relief regions, scores them with interpretable geomorphic metrics, clusters them spatially, and exports GIS-ready outputs. The project includes both a command-line workflow and a Streamlit app for running and reviewing results in one place.

MayaScan is designed for triage, not confirmation. It highlights terrain anomalies that may deserve a closer look; archaeological interpretation and field validation still require expert review.

<p align="center">
  <img src="assets/caracol_caana.png" width="700" alt="Caana pyramid at Caracol, Belize">
</p>
<p align="center">
  <em>Visible architecture at Caracol, Belize</em>
</p>

<p align="center">
  <img src="assets/aguada_fenix_lidar.png" width="700" alt="LiDAR terrain model of Aguada Fenix, Mexico">
</p>
<p align="center">
  <em>LiDAR can reveal large-scale architecture hidden beneath vegetation</em>
</p>

## Highlights

- Converts LAZ/LAS input into DTM, LRM, and density rasters
- Detects region-level candidate features instead of relying on centroid-only logic
- Supports overlap-aware multi-threshold consensus to reduce one-threshold artifacts
- Scores candidates with interpretable components such as density, relief, prominence, compactness, solidity, and area
- Uses DBSCAN to group candidates into possible settlement patterns
- Exports CSV, GeoJSON, KML, Markdown, PDF, and HTML outputs
- Includes a Streamlit review app with presets, diagnostics, labeling, comparison mode, and ZIP export

## Designed For

MayaScan is currently tuned for:

- low-relief tropical landscapes
- subtle platforms and mounds, roughly `0.3-2.0 m` of relief
- tile-by-tile exploratory analysis and ranking

## Responsible Use

- MayaScan identifies **terrain anomalies**, not confirmed archaeological sites.
- All outputs should be treated as review aids and checked by domain experts.
- Coordinate data and derived products should be handled carefully to reduce the risk of disturbance or looting.
- This repository includes a single demonstration tile at `data/lidar/sample.laz` for reproducible testing.
- The project intentionally avoids publishing curated site interpretations or sensitive location outputs.

## Installation

### Requirements

- Python `3.10+`
- PDAL installed at the system level
- Python packages from `requirements.txt`

Current package minimums:

- `numpy>=1.23`, `scipy>=1.9`, `pandas>=1.5`
- `rasterio>=1.3`, `pyproj>=3.4`, `shapely>=2.0`
- `scikit-learn>=1.2`, `matplotlib>=3.6`, `reportlab>=3.6`, `streamlit>=1.30`

Install PDAL:

- macOS: `brew install pdal`
- Ubuntu: `sudo apt install pdal`
- Windows (conda): `conda install -c conda-forge pdal`

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended sanity checks:

```bash
pdal --version
python -c "import rasterio, pyproj, scipy, streamlit"
```

## Quick Start

The repository includes a smoke-test tile at `data/lidar/sample.laz`.

### Streamlit app

```bash
streamlit run app.py
```

Then:

1. Use `data/lidar/sample.laz` or upload your own `.laz/.las` file
2. Choose a preset; `Balanced (Recommended)` is the default starting point
3. Enter a run name and click **Run MayaScan**
4. Review the map, ranked candidates, diagnostics, and score breakdown in the **Results** tab
5. Optionally compare presets or add analyst labels (`likely`, `unlikely`, `unknown`)

### CLI

Minimal run:

```bash
python maya_scan.py \
  -i data/lidar/sample.laz \
  --name caracol_sample_test \
  --overwrite \
  --try-smrf
```

Show all options:

```bash
python maya_scan.py --help
```

Outputs are written to:

```text
runs/<run_name>/
```

The HTML report is written to:

```text
runs/<run_name>/report.html
```

<details>
<summary>Advanced CLI example</summary>

```bash
python maya_scan.py \
  -i data/lidar/sample.laz \
  --name caracol_sample_test \
  --overwrite \
  --try-smrf \
  --pos-thresh auto:p96 \
  --min-density auto:p60 \
  --density-sigma 40 \
  --max-slope-deg 20 \
  --consensus-percentiles 95,96,97 \
  --consensus-min-support 2 \
  --consensus-radius-m 12 \
  --min-peak 0.50 \
  --min-area-m2 25 \
  --max-area-m2 1200 \
  --min-extent 0.38 \
  --max-aspect 3.5 \
  --edge-buffer-m 10 \
  --min-spacing-m 15 \
  --min-prominence 0.10 \
  --min-compactness 0.12 \
  --min-solidity 0.50 \
  --cluster-eps auto \
  --min-samples 4 \
  --report-top-n 30 \
  --label-top-n 60
```

</details>

## How It Works

1. **Ground model**: PDAL converts the point cloud into a DTM raster. Optional SMRF classification can be applied first.
2. **Local relief model**: MayaScan computes a multi-scale LRM by subtracting a broader smoothed surface from a finer one.
3. **Region detection**: connected positive-relief regions are extracted and cleaned up morphologically.
4. **Consensus support**: optional multi-threshold runs match regions across percentile thresholds using raster overlap, with centroid distance as a secondary guard, and keep candidates with enough support.
5. **Region metrics**: each candidate region gets area, peak relief, prominence, extent, aspect ratio, compactness, solidity, and size metrics.
6. **Density modeling**: a smoothed feature-density surface is built and sampled at the region level.
7. **Post-filtering**: regions are filtered by density, shape, slope, edge proximity, and spacing to reduce noise and duplicates.
8. **Scoring and clustering**: remaining candidates are ranked, clustered with DBSCAN, and annotated with distance to the densest member of their assigned cluster.
9. **Reporting**: the pipeline writes GIS exports, plots, reports, and run metadata for reproducibility.

## Key Parameters

### Detection

- `--pos-thresh auto:p96`
  Sets the positive-relief threshold in LRM space. Higher percentiles usually produce fewer, stronger candidates.
- `--min-density auto:p60`
  Sets the density threshold used for filtering and scoring.
- `--density-sigma 40`
  Controls how broadly the candidate-density surface is smoothed.
- `--max-slope-deg 20`
  Rejects steep regions using the q75 slope statistic.

### Consensus

- `--consensus-percentiles 95,96,97`
  Runs candidate extraction at multiple thresholds.
- `--consensus-min-support 2`
  Requires support from at least this many thresholded runs, including the primary run.
- `--consensus-radius-m 12`
  Sets the centroid-distance guard used when counting cross-threshold support; matches still require real raster overlap.
- `--no-consensus`
  Disables consensus filtering entirely.

### Shape cleanup

- `--min-peak`, `--min-area-m2`, `--max-area-m2`
  Remove features that are too weak, too small, or too large.
- `--min-extent`, `--max-aspect`
  Suppress elongated or poorly filled regions.
- `--min-prominence`, `--min-compactness`, `--min-solidity`
  Remove regions that look weak, linear, or fragmented.
- `--edge-buffer-m`, `--min-spacing-m`
  Reduce tile-edge artifacts and near-duplicate detections.

### Clustering and reporting

- `--cluster-eps auto`
  Uses automatic or fixed DBSCAN radius in meters. `auto` estimates eps from the k-distance knee with a percentile fallback.
- `--min-samples`
  Sets the minimum candidates needed to form a cluster.
- `--report-top-n`, `--label-top-n`
  Control how many candidates are emphasized in reports and KML labels.

## Outputs

Each run writes a folder under `runs/<run_name>/`. Common outputs include:

- `dtm.tif`, `lrm.tif`, `mound_density.tif`
- `candidates.csv`
- `candidates.geojson`, `candidates.kml`
- `report.md`, `report.pdf`, `report.html`
- `html/img/` candidate cutouts for the HTML report
- `plots/` diagnostic plots and histograms
- `run_params.json` with resolved settings, thresholds, accounting, and runtimes
- `candidate_labels.csv` when analyst labeling is used

The Streamlit app can also prepare a ZIP archive of run outputs. Across runs, MayaScan appends summary information to `runs/manifest.csv`.

Candidate exports include clustering fields such as `cluster_id` and `dist_to_core_km`, where `dist_to_core_km` is the distance to the densest candidate within the same cluster.

## Scoring and Run Quality

By default, candidates are ranked with this multiplicative score:

```text
Score =
  Density^1.00
  x PeakRelief^1.00
  x Extent^0.35
  x Support^0.40
  x Prominence^0.75
  x Compactness^0.20
  x Solidity^0.20
  x Area^0.50
```

The score is only meaningful within a run. It is a ranking signal, not a calibrated probability.

The Streamlit app also reports a simple run-quality heuristic based on five checks:

1. `8 <= candidates <= 250`
2. at least one non-noise cluster
3. `top_score >= 2.0`
4. `median_score >= 0.35`
5. `noise / candidates <= 0.70`

Quality badges such as `Strong`, `Moderate`, and `Weak/noisy` are meant for triage only.

## Data Sources

Public LiDAR datasets can be downloaded from [OpenTopography](https://opentopography.org/).

Typical workflow:

1. Download LAZ tiles for an area of interest
2. Place them under `data/lidar/`
3. Run MayaScan on the local files

MayaScan currently works on local input files only. No API key is required.

## Limitations

- MayaScan does not confirm archaeological features; it only prioritizes anomalies.
- Scores are relative within a run and should not be interpreted as probabilities.
- Strict consensus settings can suppress isolated true positives.
- False positives are more common in rugged terrain, modern earthworks, and heavily modified landscapes.
- Output quality depends on point-cloud quality, ground classification quality, and parameter choice.
- Analyst labels are review metadata, not training labels.
- The current workflow is mainly tuned for single-tile or tile-at-a-time analysis.

## Future Work

- multi-tile regional analysis
- linear-feature detection
- automated parameter adaptation from diagnostics and analyst feedback

## Repository Layout

```text
MayaScan/
├── app.py
├── maya_scan.py
├── README.md
├── requirements.txt
├── LICENSE
├── assets/
│   ├── mayascan_logo.svg
│   ├── caracol_caana.png
│   └── aguada_fenix_lidar.png
└── data/
    └── lidar/
        ├── .gitkeep
        └── sample.laz
```

Generated outputs under `runs/` and local LiDAR files under `data/lidar/` are typically gitignored, except for the bundled sample tile.

## Tech Stack

- Python
- NumPy, SciPy, Pandas
- Rasterio, PyProj, Shapely
- PDAL
- scikit-learn
- Matplotlib, ReportLab
- Streamlit

## Development Note

Large language models were used for prototyping, debugging support, and documentation refinement. Method choices, parameter interpretation, and project validation were reviewed manually.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

**James Adelhelm**  
Software Developer on the Data Ingest team at AccuWeather.

MayaScan is an independent personal research and software project driven by an interest in Maya history. It is not affiliated with, endorsed by, or sponsored by AccuWeather.

## Image Credits

**Caana, Caracol (Belize)**  
Photo by Devon Jones - Wikimedia Commons  
License: CC BY-SA 3.0  
<https://commons.wikimedia.org/wiki/File:Caracol-Temple.jpg>

**Aguada Fenix LiDAR**  
Courtesy of Takeshi Inomata - Wikimedia Commons  
License: CC BY-SA 4.0  
<https://commons.wikimedia.org/wiki/File:Aguada_F%C3%A9nix_1.jpg>
