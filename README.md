# MayaScan

**LiDAR triage tool for Maya archaeological survey**

MayaScan turns a raw LAZ/LAS point cloud into a ranked list of terrain anomalies for analyst review. It builds a Digital Terrain Model and a multi-scale Local Relief Model, extracts candidate positive- and negative-relief regions, scores them on shape and geomorphic metrics, clusters them spatially, and writes GIS-ready outputs (GeoJSON, KML, GeoTIFF) plus an interactive HTML report.

**MayaScan is a triage tool, not a detection system.** It produces a prioritized list of anomalies worth looking at; it does not confirm archaeological features. Expert review and field verification are always required.

---

## What it detects

MayaScan is tuned for low-relief tropical landscapes — subtle platforms, mounds, and depressions in the range of 0.3–2.0 m, on tiles with at least 4 pts/m² ground return density. A second-pass depression mode (`--detect-depressions`) looks for aguadas (reservoirs), plazas, and quarries in the same tile.

It is not designed for steep or rugged terrain, highly modified landscapes (active agriculture, modern earthworks), or sub-meter-resolution feature mapping.

---

## Responsible use

- Outputs are terrain anomalies, not confirmed sites.
- Scores are relative within a single run; they have no meaning across runs or as probabilities.
- False-positive and false-negative rates on real archaeological data are unknown.
- Handle coordinate outputs carefully. Do not publish specific location data in ways that increase looting risk.

---

## Installation

### Requirements

- Python 3.10+
- PDAL (system-level)
- Python packages listed in `requirements.txt`

**Install PDAL:**

```bash
# macOS
brew install pdal

# Ubuntu / Debian
sudo apt install pdal

# Windows (conda)
conda install -c conda-forge pdal
```

**Install Python dependencies:**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Verify:**

```bash
pdal --version
python -c "import rasterio, pyproj, scipy, streamlit; print('OK')"
```

**Full pipeline check (synthetic data):**

```bash
python tests/make_synthetic_lidar.py
python maya_scan.py \
  -i data/lidar/synthetic_mounds.las \
  --name install_test \
  --overwrite \
  --try-smrf \
  --validate-against data/synthetic_mounds_truth.csv \
  --validate-match-radius-m 8
```

A working installation recovers at least 12 of 15 synthetic mounds. See [Synthetic Validation](#synthetic-validation) for what this does and does not prove.

---

## Quick start

### Streamlit app

```bash
streamlit run app.py
```

1. Use `data/lidar/bz_lr_las47_crs.laz` (4.4 km from Caracol monumental core) or upload your own `.laz/.las` file
2. Pick a preset — **Balanced (Recommended)** is the right starting point
3. Enter a run name and click **Run MayaScan**
4. Review the map, candidate table, and score breakdown in the **Results** tab
5. Optionally compare presets or add analyst labels (`likely`, `unlikely`, `unknown`)

### CLI

```bash
python maya_scan.py \
  -i data/lidar/bz_lr_las47_crs.laz \
  --name my_run \
  --overwrite \
  --try-smrf
```

With depression detection enabled:

```bash
python maya_scan.py \
  -i data/lidar/bz_lr_las47_crs.laz \
  --name my_run_with_depressions \
  --overwrite \
  --try-smrf \
  --detect-depressions
```

Outputs land in `runs/<run_name>/`. The interactive report is at `runs/<run_name>/report.html`.

<details>
<summary>Full CLI reference example</summary>

```bash
python maya_scan.py \
  -i data/lidar/bz_lr_las47_crs.laz \
  --name my_run \
  --overwrite \
  --try-smrf \
  # --- LRM thresholds ---
  --pos-thresh auto:p96 \       # relief percentile cutoff
  --min-density auto:p60 \      # candidate-density gate
  --density-sigma 40 \          # smoothing radius for density surface
  --max-slope-deg 20 \          # reject steep terrain (q75 slope)
  # --- Consensus (multi-threshold) ---
  --consensus-percentiles 95,96,97 \
  --consensus-min-support 2 \
  --consensus-radius-m 12 \
  # --- Shape filters ---
  --min-peak 0.50 \             # minimum relief at peak pixel (m)
  --min-area-m2 25 \
  --max-area-m2 1200 \
  --min-extent 0.38 \           # bbox fill ratio [0,1]
  --max-aspect 3.5 \            # long/short axis ratio
  --min-prominence 0.10 \       # peak minus surrounding ring mean (m)
  --min-compactness 0.12 \      # 4πA/P² circularity [0,1]
  --min-solidity 0.50 \         # area / convex-hull area [0,1]
  # --- Tile edge and duplicates ---
  --edge-buffer-m 10 \
  --min-spacing-m 15 \
  # --- Clustering ---
  --cluster-eps auto \
  --min-samples 4 \
  # --- Output ---
  --report-top-n 30 \
  --label-top-n 60
```

</details>

---

## How it works

1. **Point cloud QC** — PDAL reports estimated ground-return density. MayaScan warns below 4 pts/m² and aborts below 1 pt/m²; DTM quality degrades below these thresholds.

2. **Ground model** — PDAL builds a DTM at the requested resolution (default 1 m/px). If `--try-smrf` is set, PDAL's SMRF classifier runs first to isolate ground returns from unclassified or vegetation-heavy data. SMRF parameters (`--smrf-scalar`, `--smrf-slope`, `--smrf-threshold`, `--smrf-window`) can be tuned for non-standard terrain.

3. **Local Relief Model (LRM)** — The LRM isolates micro-topographic relief by subtracting a broadly Gaussian-smoothed surface from a narrower one. MayaScan does this across multiple scale pairs and retains the maximum at each pixel, preserving the strongest relief signal regardless of which scale best matches a given feature. Default sigma pairs (small: 1–2 px; large: 8–16 px) are tuned for features 5–200 m² at 1 m resolution, covering typical Maya platform and mound sizes. The max-fusion operator is not published in peer-reviewed literature; users should cross-check results against known features before treating output as authoritative.

4. **Region detection** — Pixels above the relief threshold are extracted as connected regions and cleaned up morphologically (hole-fill, small-component removal).

5. **Consensus filtering** — If enabled, the same detection runs at multiple LRM percentile thresholds. A candidate is kept only if it appears in at least N of those runs, determined by real raster overlap. This reduces one-threshold noise but can suppress isolated true positives.

6. **Region metrics** — Each region is measured: area, peak relief, local prominence (peak minus surrounding ring mean), bounding-box extent, aspect ratio, compactness (4πA/P²), and solidity (A / convex-hull area).

7. **Candidate-density surface** — A smoothed density raster is built from the spatial distribution of candidate centroids across the tile and sampled per region. This gives a bonus to features in spatially coherent groups and penalizes isolated outliers.

8. **Shape and density filters** — Candidates are dropped by minimum/maximum thresholds on all the above metrics, plus slope, tile-edge proximity, and centroid spacing. Remaining candidates are scored and ranked.

9. **Depression detection (optional)** — With `--detect-depressions`, the LRM is inverted and the same pipeline runs again. Depression candidates (aguadas, plazas, quarries) are appended to the output with `feature_type=depression` and shown as dashed blue markers in the report. They are not included in mound clustering.

10. **Clustering** — Mound candidates are clustered with DBSCAN using an automatic or fixed radius. Each candidate is annotated with its cluster ID and distance to the densest member of its cluster.

11. **Reporting** — Every run writes a time-stamped folder with GIS exports, diagnostic plots, a `run_params.json` for reproducibility, and an HTML report with an interactive Leaflet map.

---

## Outputs

```
runs/<run_name>/
├── dtm.tif                      DTM raster
├── lrm.tif                      Local Relief Model (NaN where DTM is void)
├── mound_density.tif            Smoothed candidate-density surface
├── candidates.csv               All candidates with metrics and scores
├── candidates.geojson           Centroids as GeoJSON Points (WGS84)
├── candidate_regions.geojson    Bounding-box Polygons (WGS84) — load in QGIS
├── candidates.kml               KML for Google Earth, labeled by rank
├── report.html                  Interactive Leaflet report (open in browser)
├── report.md / report.pdf       Plain text and PDF summaries
├── plots/                       Density overlay, score histograms, etc.
├── html/img/                    LRM cutouts for each top candidate
└── run_params.json              All resolved settings, thresholds, runtimes
```

`candidate_labels.csv` is written when analyst labeling is used via the Streamlit app.

`runs/manifest.csv` accumulates one summary row per run across the project.

### Coordinate accuracy

All output coordinates are WGS84 (EPSG:4326). Reported positions are region centroids. Horizontal accuracy is typically ±2–5× the input resolution and degrades further for large or irregular regions. Do not use these coordinates for field navigation or sub-meter survey work.

---

## Scoring

Candidates are ranked by this multiplicative score:

```
Score =
  Density^1.00
  × PeakRelief^1.00
  × Extent^0.35
  × ConsensusSupport^0.40
  × Prominence^0.75
  × Compactness^0.20
  × Solidity^0.20
  × Area^0.50
```

The score is meaningful only for ranking within a single run. It is not a probability, not comparable across runs, and the exponents are heuristic — they have not been calibrated against confirmed archaeological features.

The Streamlit app shows a **Triage quality** badge based on five parameter-output checks (candidate count in range, at least one cluster, score thresholds, noise fraction). This badge reflects whether the run's parameter settings produced a plausible output; it says nothing about whether real features are present.

---

## Key parameters

### Where to start

Use the **Balanced** preset and accept its defaults for a first pass. Review the HTML report map and candidate table. Adjust only if the results are clearly too sparse or too noisy.

### Detection thresholds

| Flag | Default | Effect |
|---|---|---|
| `--pos-thresh auto:p96` | 96th LRM percentile | Raise to get fewer, stronger candidates. Lower to increase recall at cost of more noise. |
| `--min-density auto:p60` | 60th density percentile | Lower if the tile is sparse or has few candidates. |
| `--density-sigma 40` | 40 pixels | Controls how broadly candidate density is smoothed. |
| `--max-slope-deg 20` | 20° (q75 slope) | Lower in very flat terrain; raise in sloped or hilly areas. |

### If you get too few candidates

Try `--pos-thresh auto:p95` and `--min-density auto:p50`. Relax shape filters: lower `--min-compactness` to 0.08, `--min-solidity` to 0.40, `--min-extent` to 0.30. If the tile is vegetation-dense, use `--try-smrf`.

### If you get too many candidates (noisy output)

Raise `--pos-thresh auto:p97`. Increase `--min-peak` to 0.60, `--min-compactness` to 0.16, `--min-solidity` to 0.58. Increase `--consensus-min-support` to 3.

### Shape filters

| Flag | Purpose |
|---|---|
| `--min-peak 0.50` | Minimum relief at the peak pixel (m). |
| `--min-area-m2 25` / `--max-area-m2 1200` | Size bounds. Increase max for large platforms. |
| `--min-extent 0.38` | Bbox fill ratio [0,1]; suppresses elongated blobs. |
| `--max-aspect 3.5` | Long/short axis ratio; suppresses ridges and walls. |
| `--min-prominence 0.10` | Peak minus surrounding-ring mean (m). |
| `--min-compactness 0.12` | Circularity 4πA/P² [0,1]; suppresses linear features. |
| `--min-solidity 0.50` | Area / convex-hull area [0,1]; suppresses fragmented shapes. |
| `--edge-buffer-m 10` | Drop regions within this distance of the tile edge. |
| `--min-spacing-m 15` | Score-ordered deduplication radius. |

### Consensus

| Flag | Purpose |
|---|---|
| `--consensus-percentiles 95,96,97` | Threshold levels to run. |
| `--consensus-min-support 2` | Minimum number of runs that must agree. |
| `--consensus-radius-m 12` | Centroid-distance guard (overlap is the primary criterion). |
| `--no-consensus` | Disable consensus entirely for a single-threshold run. |

### Clustering

| Flag | Purpose |
|---|---|
| `--cluster-eps auto` | DBSCAN radius, estimated from k-distance knee. Use a fixed value (meters) if auto gives bad results. |
| `--min-samples 4` | Minimum candidates to form a cluster. |

---

## Synthetic validation

`tests/make_synthetic_lidar.py` generates a controlled point cloud: 500×500 m at UTM 16N, with 15 Gaussian mounds at known positions (amplitudes 0.33–0.62 m, σ = 3–6 m) and 2 elongated ridge traps that should not be detected.

```bash
python tests/make_synthetic_lidar.py
python maya_scan.py \
  -i data/lidar/synthetic_mounds.las \
  --name synth_validation \
  --overwrite \
  --try-smrf \
  --validate-against data/synthetic_mounds_truth.csv \
  --validate-match-radius-m 8
```

With default Balanced parameters: **14/15 mounds detected (93% recall), 0 false positives** from the ridge traps. The missed mound (M6) is adjacent to a long ridge whose shape leaks into M6's bounding box, causing it to fail the compactness filter — this is correct behavior.

**What this proves:** The detection pipeline is internally consistent and the shape filters effectively suppress elongated features.

**What this does not prove:** Performance on real archaeological data. The synthetic mounds were sized to match the default parameters. False-positive and false-negative rates on real sites are unknown and must be measured against confirmed features before operational use.

---

## Data sources

Public LiDAR for Maya lowlands can be downloaded from [OpenTopography](https://opentopography.org/). Download LAZ tiles, place them under `data/lidar/`, and run MayaScan locally. No API key is required.

---

## Limitations

- MayaScan detects terrain anomalies, not confirmed sites.
- Scores are meaningless across runs and should not be treated as probabilities.
- False-positive rates are elevated in karst topography, modern earthworks, tree-throw mounds, and heavily disturbed landscapes.
- Consensus filtering reduces noise but can suppress isolated mounds, which are common in dispersed settlement zones.
- DTM quality depends on ground classification. Poor SMRF results produce poor LRMs.
- Results are per-tile; features near tile edges or split across tiles may be missed or duplicated.
- **False-positive and false-negative rates on real archaeological data have not been quantified.**
- **The multi-scale LRM max-fusion operator is novel and unvalidated in peer-reviewed literature.**
- **Scoring exponents are heuristic** and have not been calibrated against confirmed features.
- Coordinate centroids carry ±2–5× resolution horizontal uncertainty.
- If the input tile spans a UTM zone boundary, clustering distances and coordinate exports may be slightly inaccurate.

---

## Repository layout

```
MayaScan/
├── app.py                   Streamlit review app
├── maya_scan.py             CLI pipeline
├── README.md
├── requirements.txt
├── LICENSE
├── assets/
│   ├── mayascan_logo.svg
│   ├── caracol_caana.png
│   └── aguada_fenix_lidar.png
├── tests/
│   └── make_synthetic_lidar.py   Generates synthetic tile for validation
└── data/
    └── lidar/
        ├── .gitkeep
        └── bz_lr_las47_crs.laz   Default tile — 4.4 km from Caracol monumental core
```

`runs/` and `data/lidar/*.las/.laz` are gitignored.

---

## Tech stack

| Library | Role |
|---|---|
| PDAL | Point cloud ingestion, SMRF ground classification, DTM rasterization |
| NumPy / SciPy | Raster arithmetic, Gaussian filtering, morphological ops, clustering |
| Rasterio | GeoTIFF read/write with CRS and affine transform support |
| PyProj / Shapely | CRS reprojection, WGS84 coordinate conversion, polygon geometry |
| scikit-learn | DBSCAN spatial clustering |
| Pandas | Candidate tabulation and CSV export |
| Matplotlib | Diagnostic plots and LRM cutouts |
| ReportLab | PDF report rendering |
| Streamlit | Interactive review and labeling app |
| laspy | LAS file generation for synthetic test harness |

---

## Images

<p align="center">
  <img src="assets/caracol_caana.png" width="600" alt="Caana pyramid at Caracol, Belize">
</p>
<p align="center"><em>Caana pyramid at Caracol, Belize — the type of architecture MayaScan is designed to help locate in dense forest</em></p>

<p align="center">
  <img src="assets/aguada_fenix_lidar.png" width="600" alt="LiDAR terrain model of Aguada Fenix, Mexico">
</p>
<p align="center"><em>Aguada Fénix revealed by LiDAR — a large Maya reservoir invisible on the ground surface</em></p>

---

## Development note

Large language models were used for prototyping, debugging, and documentation. Method choices, parameter interpretation, and validation were reviewed manually.

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

## Author

**James Adelhelm**  
Software Developer, Data Ingest team, AccuWeather

MayaScan is an independent personal research project driven by interest in Maya history. It is not affiliated with, endorsed by, or sponsored by AccuWeather.

---

## Image credits

**Caana, Caracol (Belize)** — Devon Jones, Wikimedia Commons, CC BY-SA 3.0  
<https://commons.wikimedia.org/wiki/File:Caracol-Temple.jpg>

**Aguada Fénix LiDAR** — Takeshi Inomata, Wikimedia Commons, CC BY-SA 4.0  
<https://commons.wikimedia.org/wiki/File:Aguada_F%C3%A9nix_1.jpg>
