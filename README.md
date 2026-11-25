# Skyliner

ML skyline annotator for photos taken at sea. It produces the **Horizon contract v2 observation**
(`horizon.observation/2+json`) plus every horizon level the requirements ask for, per frame, in the
same folder layout as `example_dataset/`, with more detail.

Requirements: `SKYLINER_REQUIREMENTS.md`. Experiment log: `docs/development_process/`.
Out-of-fold accuracy on the 116 labelled photos: `reports/cv_v0.2/report.md` (v0.1: `reports/cv_v0.1/`).

## Run (CPU, no GPU, no torch)

```bash
pip install -r requirements.txt            # numpy, opencv-python, onnxruntime, pillow
python -m skyliner annotate example_dataset annotated_dataset
python -m skyliner score annotated_dataset example_dataset --report report.md --csv per_photo.csv
```

Input: a folder of frames (`<id>/photo.jpg` + `meta.json` with `width, height, focal_px, time`,
optional `gps.json`), a folder of JPEGs, or one JPEG. Without `meta.json` the focal length comes from
EXIF (`FocalLengthIn35mmFilm`) times the camera profile in `skyliner/sensors.py`, or `--focal-px`.

Useful flags: `--platform ferry`, `--eye-height 2` (m above sea; sets the dip), `--utc-offset 3`
(photo clock is local time), `--with-truth` (copy `gps.json` into `gnss_truth`, evaluation only),
`--emit-low-quality`, `--strict-horizon` (see below), `--copy-photo`, `--no-overlay`.

## Output per frame (`annotated_dataset/<id>/`)

| file | content |
|---|---|
| `observation.json` | contract v2: `id, platform_id, sensor_id, t_start_ns, t_end_ns, frame, az_deg, elev_deg, state, sigma_deg, eye_height_m, roll_deg, pitch_deg, source` + PROPOSED §8: `layers` (2 × `inner_ridge`, 1 × `waterline`; NaN = absent), `lower_bound`, `visibility_km` (NaN, not estimated yet). Not written when no sea is in view (no attitude). |
| `label.json` | hand-label format (2048 grid, step 4): `horizon, outer, inner1, inner2, occl` + `waterline, waterline_sigma, occl_type, horizon_quality, state, reason, outer_sigma, outer_raw, sigma_deg, lower_bound, horizon_seen, contrast, source` |
| `meta.json` | `width, height, focal_px, horizon_a, horizon_b, name, time` + `roll_deg, pitch_deg, horizon_source, horizon_quality, horizon_sigma_px, horizon_points, attitude, obstacles (count per type), waterline_columns, eye_height_m, counts, timings_s` |
| `gps.json` | copied from the input if present |
| `obstacle_mask.png` | photo-resolution obstacle mask (255 = obstacle: own ship, vessels, clouds, glare, wake/spray) |
| `obstacles.json` | obstacle instances: `id, type, bbox_px, area_frac, mean_value, mean_saturation, touches_edges` |
| `overlay.jpg` | QA view: magenta obstacles with their type, cyan L0 (yellow-green if low quality), yellow L3 waterline, red L1, orange L1 flagged `lower_bound`, green/magenta inner ridges; top bar per column: blue sea, red land, grey unknown, magenta obstacle |

`sigma_deg` is `Infinity` for unknown samples and layer arrays use `NaN` for "absent", as the contract
defines. Python's `json` reads both.

## How it works

Two small CNNs, exported to ONNX:

1. **SceneNet** (512×384): per-pixel sky / land / sea / occluder. It decides the topology: which
   boundary is the topmost land, where open sea shows, where ship structure, wake or glare hides the view.
2. **EdgeNet** (full resolution, run only on the skyline band): sub-pixel heatmaps for L0 (visible
   sea horizon), L1 (silhouette) and L2 (inner ridges).

Post-processing (`skyliner/infer.py`), per label column (512 columns, ≤ 4 px spacing):
- **L0**: a coarse RANSAC line through the SceneNet sky/land→sea boundaries. It is refined on the
  EdgeNet L0 heatmap (open sea) and the channel-3 heatmap (sea line under distant land), then a robust
  IRLS line gives roll and pitch (with dip), per §4.2.
- **L1**: topmost-wins. The topmost land boundary from SceneNet, then the topmost strong EdgeNet peak
  near it.
- **State** per §5. `sea` only where the open-sea edge is actually seen. `unknown` for occluders on the
  skyline band, land reaching the frame top, no edge, glare or spray, and isolated runs of fewer than 3 columns.
- **σ**: heatmap spread → calibrated on out-of-fold height errors, per horizon source (`models/calibration.json`).
- **lower_bound**: SceneNet sees faint land above the traced line (haze). σ is inflated and the flag is set.
  Very faint land (contrast < 0.025, or haze-flagged with contrast < 0.10) becomes `unknown`.
- **L2**: EdgeNet ridge peaks between L1 and L0, chained across columns (≤ 3 px steps); chains < 10 columns are dropped.

## Obstacles, waterline, horizon quality (no rejections)

- **Obstacles** are segmented by SceneNet (one class: anything not in the DEM; trained on the labelled
  ship structure plus pasted occluders, synthetic poles and railings, and synthetic clouds on
  summits). They are masked out of the horizon fit, and a skyline column touched by one is
  `unknown` with `reason: "obstacle: <type>"`. Instances are typed heuristically (geometry and
  colour): `ship_structure, vessel_object, cloud, glare, wake_spray, other`.
- **Waterline (L3)**: where land meets the sea, measured in every column where it is seen (EdgeNet
  channel 3 near the SceneNet land→sea boundary). It is a layer in the observation and `waterline` in
  `label.json`. With no open-sea horizon (`horizon_source: "waterline"`), L0 is the line through it.
- **Horizon quality**:
  - `good`: refined fit on ≥ 24 points over ≥ 25 % of the width, within 2° of the coarse line,
    residual ≤ 5 px.
  - `low`: weaker evidence, e.g. the sea line is mostly behind machinery. The best line is kept, and σ
    gets an offset term (≥ 3 % of the frame height) plus a 3° roll term that grows away from the evidence.
  - `none`: no sea in view. The frame is annotated in pixels (L1, obstacles, overlay) and no
    observation is written.
- Low-quality frames get the full annotation. Their observation is written as
  `observation.low_quality.json`, so it doesn't reach Horizon, which weighs σ only inside a 0.5° bin
  today. `--emit-low-quality` writes it as `observation.json`. Waterline frames where land rises
  > 4° (near shore) are `low` too: a near waterline sits below the true horizon. `--strict-horizon` rejects frames without an open-sea
  horizon (requirements §4.2).

## Training (needs torch; done on a CPU box)

```bash
python -m skyliner.train --data example_dataset --net scene --fold all --steps 1100
python -m skyliner.train --data example_dataset --net edge  --fold all --steps 1300
python -m skyliner.export --runs runs --fold all --out models
# 4 contiguous folds (+ calibration + report):
for k in 0 1 2 3; do python -m skyliner.train ... --fold $k; done
python -m skyliner.crossval --data example_dataset --runs runs --folds 4 --work cv --models models
```

Label corrections used for training and scoring are in `skyliner/label_fixes.json` (occluder polygons,
untrusted columns). See the development log for why.

## Not done yet

- `.hzo` (FlatBuffers) output and MQTT publishing: they need `horizon_contract` and the schema from the
  geolocator repo. The JSON form is complete.
- L3 waterline, `visibility_km`: no ground truth exists yet (§10). The fields are reserved.
- End-to-end localization (`scripts/17_locate.py`): runs in the Horizon repo, on this output.
