# Skyliner: requirements for the skyline annotator that feeds Horizon

The skyliner is a separate system. It takes photos taken at sea and annotates every horizon level in them. Its output is the only input of the Horizon geolocator. This document specifies that output and its quality, so that Horizon can get the most out of it.

Status of each requirement:
- **MUST**: Horizon uses it today (contract v2). Without it the photo is not localized, or is localized worse.
- **SHOULD**: Horizon uses it today. It improves the result or the system's behaviour.
- **PROPOSED**: an additive contract extension. Horizon does not read it yet, and each one lists the core work it needs. A skyliner that produces it now is ready the day the core uses it. Old cores ignore it safely.

The references are `contract/schema/horizon.fbs` (fields, units, defaults) and `contract/README.md` §4 (rules). Where this document and the schema disagree, the schema wins. A change to the schema is agreed with Fotis first.

---

## 1. What Horizon does with the output

These are the constraints that every requirement below comes from.

1. **Binning.** Horizon turns the samples into **0.5° azimuth bins** (`observations.to_bins`).
   - A bin's elevation is the σ-weighted mean of its known samples.
   - A bin is usable only if **at least half of its samples are known** (`sea` or `land`, with a finite σ).
2. **Gaps.** Before matching (`queries.clean_query`), unusable bins are trimmed at both ends.
   - Interior gaps of **≤ 10° (20 bins)** are bridged by linear interpolation.
   - At a longer gap the query is split, and only the **longest clean stretch** is kept.
   - Fewer than **4° (8 bins)** of usable skyline means no answer.
3. **Matching.** The query is matched against about 2 M precomputed terrain profiles, the **topmost skyline** (silhouette) seen from each grid point.
   - The profiles are rendered from Copernicus GLO-30 (30 m surface model) for an eye **2 m** above the sea.
   - The renderer uses refraction **k = 0.13**, ranges **200 m – 230 km**, and earth curvature.
   - The matcher searches over every heading, so the absolute azimuth is never needed.
   - The transformer retriever (always on) embeds the same 0.5° bins.
4. **Tracking.** Observations are grouped per `platform_id` into bursts by time. A gap of more than 120 s starts a new burst.
5. **Not used.** Horizon never looks at pixels, and never reads GNSS (`gnss_truth` is for evaluation only).

What follows from this:
- **The silhouette carries almost all the information today.** Every inner level (§3) is extra information that the current core cannot use yet.
- **A wrong skyline is worse than a missing one.** Marking a sample `unknown` costs little: it is bridged or trimmed. Tracing the wrong ridge, or calling haze land, puts a false shape into the match.
- **Width matters.** More continuous azimuth gives a more unique shape. A 360° stitched sweep is the best case.

---

## 2. Measured reference points

| | Result |
|---|---|
| Hand-drawn labels (116 ferry photos, iPhone 13) | 91–92 / 116 within 1 km, median error ~265 m |
| Hand labels + transformer retriever | 96 / 116 |
| Current learned extractor (same photos) | 78 / 110 within 1 km |
| Angular resolution | 1 px ≈ 0.036° at 2048 px width, 26 mm equivalent |
| Skyline error that still localizes like the labels | ≈ **0.06° at p90** (≈ 1.7 px) |
| Effect of a 3.1 % focal-length error | grows linearly to **0.4°** at the photo edges, which is far beyond the 0.06° budget |

**System target:** on the same photos, the skyliner's observations localize **at least as well as the hand labels**, with **no more confident-wrong answers** (confident and > 5 km off) than the labels.

---

## 3. The horizon levels to annotate

These match the hand-label format (`data/skyline_labels/README.md`): the `horizon` row, the land layers `outer` / `inner1` / `inner2`, and the `occl` mask. Per image column, from top to bottom:

```
 sky
 ───────╮      ╭──────────╮               L1 silhouette: topmost land/sky boundary (far ridge if it rises above a near one)
        ╰──╮╭──╯ far      ╰────╮
   near ───╯╰─────╮            ╰──        L2 inner ridges: occluding contours below L1 (a nearer ridge in front of a farther one)
 ═════════════════╧═══════════════════    L0 sea horizon: open-sea line (the level reference)
        ▔▔▔▔▔▔▔▔▔▔▔                       L3 waterline: where a near land mass meets the sea, BELOW L0
 sea
 [####]                                   occluders: ship structure, other vessels, spray, glare, cloud on land -> unknown
```

| Level | What it is | Used by Horizon | Status |
|---|---|---|---|
| **L0** sea horizon | The open-sea line. It gives roll, pitch and the elevation zero (§4.2). Where it is visible with no land above it, those samples are `sea`. | Yes: the angle reference, and the `sea` samples | **MUST** |
| **L1** silhouette | The topmost boundary between land and sky in every column. This is what the database stores. | Yes: matched | **MUST** |
| **L2** inner ridges | Visible ridge lines below L1: the outline of a nearer ridge or island against a farther land mass. Up to 2 per column, as in the labels. | No | PROPOSED (§8.1) |
| **L3** waterline | Where land meets the sea, seen below L0 for near land. Its depression angle encodes the range to that shore. | No | PROPOSED (§8.1) |
| occluders | Anything that hides L0 or L1 and is not in the DEM: own ship, masts, railings, other vessels, buoys, cranes and harbour structures nearer than 200 m, people, birds, spray, glare, cloud or fog on the land | Yes: as `unknown` | **MUST** |

Rules for the silhouette (L1):
- **Topmost always wins.** A faint, hazy far ridge that rises above a sharp near ridge **is** the silhouette. Tracing the near ridge there is the single most damaging error the current extractor makes (`extractor/DESIGN.md` §2.2).
- **Not visible means unknown, not lower.** If haze may hide farther land above what is traced, the traced line is a lower bound. Mark it `unknown`, or `land` with a large σ, or use the PROPOSED `lower_bound` flag (§8.2). Never present it as an exact silhouette with a small σ.
- **Cloud.** Cloud sitting on a summit makes those samples `unknown`. Never trace the cloud's outline. Clouds that are clearly above the land and do not touch it are ignored.
- **Built-up land.** Buildings, forest and towers on land farther than about 1 km are roughly part of a 30 m surface model, so trace the outline as seen. Thin objects (masts, wind turbines, antennas, cranes) are not in it: interpolate the terrain under them, or mark them `unknown`.
- **Sea, not land.** Between land masses, where the open-sea horizon shows, the samples are `sea` at the dip angle. They are never "low land" and never left out.

---

## 4. Output: contract v2 observation

### 4.1 Fields (`horizon.v2.Observation`)

| Field | Requirement | Status |
|---|---|---|
| `id` | Unique, stable string. 1–64 characters from `A-Z a-z 0-9 _ . -` (`topics.check_id`). | **MUST** |
| `platform_id` | One per vessel or sensor mount. Tracking is per platform. | **MUST** for tracking |
| `sensor_id` | Camera or sensor identifier (it matches a sensor profile) | SHOULD |
| `t_start_ns`, `t_end_ns` | Exposure time, **UTC**, in ns since the epoch, with an error of at most ±1 s. For a single frame, `t_end_ns = t_start_ns`. Use 0 only if the time is truly unknown, because tracking needs it. | **MUST** |
| `frame` | `Camera` (relative to the optical axis) or `Platform` (relative to the bow, for frames stitched with encoder or IMU angles) | **MUST** |
| `az_deg[]` | Strictly increasing, spanning **< 360°**. A full sweep is 0…359.9 and is not wrapped. Any spacing is allowed; native resolution (per column, or every ≤ 4 px) is preferred. | **MUST** |
| `elev_deg[]` | L1 elevation per sample, in degrees above the **true horizontal** at the camera (§4.2). For `sea` samples, this is the measured sea horizon (≈ the dip). | **MUST** |
| `state[]` | `sea` / `land` / `unknown`, one per sample (§5) | **MUST** |
| `sigma_deg[]` | 1-σ per sample, calibrated (§6). `+inf` means unknown. | SHOULD (today it only weights samples within a bin) |
| `eye_height_m`, `eye_height_sigma_m` | Camera height above the sea, if known (deck height plus mount) | SHOULD. The core matches at 2 m today and does not fit the height. |
| `roll_deg`, `pitch_deg` | The attitude used for the pixel → angle conversion | SHOULD (diagnostics, overlays) |
| `heading_hint_deg`, `_sigma_deg` | Only from a trusted compass or INS. Leave it unset otherwise, because the core never needs it. | optional |
| `gnss_truth` | Evaluation and testing only. Never required, and never used for localization. | optional |
| `source` | Skyliner name, version and model id | **MUST** (traceability) |

### 4.2 Angles: the exact convention

The convention is the same as `contract/README.md` §1 (camera model) and `extractor/DESIGN.md` §1.

- **Pinhole model.** `cx = (width−1)/2`, `cy = (height−1)/2`, `f = focal_px`.
- **Sea horizon line** `v = a·u + b`.
  - Roll = atan(a).
  - Pitch = atan2((a·cx + b − cy)·cos roll, f) + dip.
  - The pixel is de-rolled, then rotated by the pitch. Azimuth offset = atan2(x, fwd); elevation = atan2(up, hypot(x, fwd)).
- **Dip** is the sea horizon's angle below the horizontal: dip = −√(2h(1−k)/R), with k = 0.13 and R = 6 371 km. That is −0.042° at h = 2 m and −0.144° at 20 m. Use the dip of the real eye height when it is known (and report `eye_height_m`), otherwise use 2 m.
- **Focal length** must be calibrated per camera model and zoom.
  - Wrong scale tilts the skyline linearly with azimuth. The iPhone 13 needed ×1.031 over EXIF.
  - The requirement is a residual scale error of **≤ 0.3 %**. Scaling the measured 3.1 % → 0.4° gives about 0.04° at the edges, inside the 0.06° budget.
- **Lens distortion** must be corrected where it is significant. It was not significant on the iPhone 13 main camera; other lenses must be checked.
- **Refraction and curvature.** Do not correct them. Report the angles as observed. The database already models k = 0.13.
- **No open-sea horizon** (near shore, fog bank). Pitch then has no visual reference.
  - With an IMU attitude: use it, set `roll_deg` / `pitch_deg`, and inflate σ by the IMU's attitude error.
  - Without one: do not emit the observation. Report it as rejected (§7).

### 4.3 Transport

- **MQTT** (Mosquitto, QoS 1, the client keeps a persistent session).
  - Observations go to `horizon/v1/<platform>/observation`, as a binary `.hzo` (`HZO2`, the production form) or as JSON `horizon.observation/2+json`. JSON is for tests and debugging.
  - Optional, for viewers only (Horizon ignores them): the JPEG on `horizon/v1/<platform>/image/<id>`, and its camera model on `…/image/<id>/meta`. The meta is JSON with `width, height, focal_px, horizon_a, horizon_b`, scaled to the JPEG sent.
  - `horizon/v1/<platform>/reset` clears that platform's track (new voyage).
- **Files.** Use `.hzo` written with `horizon_contract.write_observations`. The v1 `.npz` is read-only legacy and is not for new work.
- **Libraries.** Use `horizon_contract` (Python) or the generated C++ header. Do not hand-roll FlatBuffers.

---

## 5. Per-sample state rules (MUST)

| Situation in a column | `state` | `elev_deg` | `sigma_deg` |
|---|---|---|---|
| Open-sea horizon, no land above | `sea` | the measured horizon angle (≈ dip) | horizon σ |
| Land against the sky, outline clear | `land` | L1 | per §6 |
| Land, outline hazy or faint but certain to be the top | `land` | L1 | inflated |
| Land traced, but farther land may be hidden in haze above it | `unknown` (or `land` + large σ + PROPOSED `lower_bound`) | traced line | large |
| Two candidate ridges, cannot tell which is the top | `unknown` (PROPOSED: alternatives, §8.3) | best guess | large |
| Own ship, rigging, railing, another vessel, buoy, person, bird | `unknown` | anything finite | `+inf` |
| Glare or sun-glitter hiding the horizon or land | `unknown` | anything finite | `+inf` |
| Cloud or fog touching the land | `unknown` | anything finite | `+inf` |
| Land closer than 200 m, or a harbour structure | `unknown` | anything finite | `+inf` |
| Sea horizon hidden by waves or spray, with no land | `unknown` | anything finite | `+inf` |

Hard rules:
- `unknown` is **never** sea. Do not fill hidden columns with the sea level: that draws a false cliff into the shape.
- **False land is worse than missed land.** At thresholds, prefer `unknown` to a guess.
- Per 0.5° bin, at least half of the samples must be known for the bin to count. A sparse scatter of known samples inside a hidden area is wasted.

---

## 6. Accuracy and σ (MUST unless noted)

They are measured against the hand labels, leave-one-photo-out, with `extractor/src/skyline_extractor/labels.score`.

| Metric | Requirement |
|---|---|
| Sea horizon row | median error ≤ **3 px** (≈ 0.1° at 1 px = 0.036°) |
| Horizon tilt (roll) | ≤ 0.05° (a tilt adds a linear error across the frame) |
| Silhouette height above the horizon, where both see land | median ≤ 0.03°, **p90 ≤ 0.06°** |
| Land recall (labelled land traced as usable land) | ≥ **0.90** |
| False land (labelled sea called land, photos with ≥ 10 sea columns) | ≤ **0.05** |
| Wrong ridge layer (L1 traced on a lower labelled layer) | ≤ 2 % of land columns |
| Photos rejected | as few as possible. A rejection must always give a reason. |
| End to end (`scripts/17_locate.py` on the 116 labelled photos) | within 1 km ≥ the hand-label result; confident-wrong (> 5 km) ≤ the hand-label count |

**σ calibration** (SHOULD).
- σ must be the real 1-σ error versus the labels: about 68 % of |error| ≤ σ, and about 95 % ≤ 2σ, checked per state and per haze level.
- A σ that is too small on hazy land is the dangerous case.
- Today σ weights samples inside a bin, and `+inf` means unknown. Matcher weighting by σ is planned, so σ quality will matter more.

---

## 7. Behaviour and operations

- **One observation per frame**, or one per stitched sweep (`Platform` frame). Stitching gives a wider span and is preferred where the mount allows it.
- **Bursts.** Keep emitting during a burst. Several frames a few seconds apart improve tracking. The time gap that splits bursts is 120 s.
- **Rejections.** When a frame is unusable (dark, no reference, all occluded), publish nothing to `observation`. Log the reason; a status topic for that is to be agreed.
- **Latency.** For now: ≤ 1 s per frame on the target CPU, to be agreed. Horizon itself takes about 3.4 s for a cold start and about 0.9 s when tracking (4 threads).
- **Privacy.** Photos and GNSS stay private. Only observations (numbers) may leave the machine.
- **Versioning.** `source` carries the version. A model change that moves the metrics in §6 is re-measured and recorded before Horizon adopts it. Horizon also adopts the skyliner's error model (`contract/README.md` §2) only on request.

---

## 8. PROPOSED extensions: using every level

All of these are additive fields at the end of `Observation`, so old cores ignore them (FlatBuffers rule). The skyliner SHOULD produce §8.1 and §8.2 from day one, because they cost little once L1 is right.

### 8.1 Layers (L2 inner ridges, L3 waterline)

```fbs
enum LayerKind : byte { InnerRidge = 0, Waterline = 1 }
table Layer {
  kind: LayerKind;
  elev_deg: [float] (required);   // aligned with Observation.az_deg; NaN = layer absent in that sample
  sigma_deg: [float];
}
// appended to Observation:
//   layers: [Layer];
```

- **Inner ridges** give a second, independent shape: the outline of nearer land against farther land.
  - **Core work:** render occluding contours, not only the maximum, along each ray. This needs new database profiles or on-the-fly rendering at the refinement stage.
  - **Value:** separates candidate positions that have the same silhouette, and pins the range to the near ridge.
- **Waterline** gives range. A point on the sea at distance d appears at a depression of about −(h/d + d(1−k)/2R) radians, so with a known eye height the waterline angle gives the distance to that shore.
  - **Core work:** a range check at refinement (the DEM coastline at that azimuth).
  - **Value:** strongest on near-shore photos, exactly where the silhouette is short or ambiguous.

### 8.2 Haze lower bound and visibility

```fbs
//   lower_bound: [bool];     // aligned with az_deg: true = the real silhouette is at or ABOVE elev_deg (haze may hide farther land)
//   visibility_km: float = nan;  // estimated meteorological visibility
```

- **Core work:**
  - a one-sided cost: penalize only rendered terrain *below* the traced line;
  - capping the render range at `visibility_km` (refinement stage, with the sector renderer).
- **Value:** turns the dominant failure (a far ridge lost in haze) from a wrong shape into a valid constraint.
- This is a separate vector and not a new `SampleState` value, because older cores treat any state other than `unknown` as exact.

### 8.3 Alternatives for ambiguous sectors

```fbs
//   alternatives: [Observation];  // the same frame, different L1 hypotheses (e.g. near vs far ridge, two horizons)
```

- **Core work:** localize each alternative and keep the answer with the best margin. This is `extractor/DESIGN.md` §2.3 option 2.
- Until it exists, emit the best hypothesis and mark the ambiguous sector `unknown`.

---

## 9. Acceptance checklist

1. `python -m pytest tests/test_contract_v2.py`, and `extractor/tests/test_contract.py` on the skyliner's own output, pass. The output is read by `horizon_contract.read_observations` and by the C++ `read_example`, with verification.
2. `example_observations.hzo` round-trips through the skyliner's writer unchanged.
3. §6 metrics on the labelled photos (leave-one-photo-out), written to a report with per-photo rows.
4. End to end: `.hzo` → `scripts/17_locate.py --with-truth` → within-1 km and confident-wrong counts versus the hand labels on the same photos.
5. The live path: publishing to `horizon/v1/<platform>/observation` gives a fix on `…/fix` and a pin in the demo UI (`DEMO.md`).
6. Error model `extractor_errors.json` (`contract/README.md` §2) produced for Horizon's benchmarks.

## 10. Ground truth for training and scoring

- Hand labels in `data/skyline_labels/` (private): for every 4 px column, the `horizon` row, the layers `outer` / `inner1` / `inner2` (None = no land), and `occl`.
- New labelling SHOULD add:
  - the waterline (L3);
  - occluder masks (no labelled photo has any yet);
  - a "haze-limited" flag per column.

  Without these the skyliner cannot learn §5 and §8.
- More labelled photos from other conditions are needed: different haze, sun angle, sea state, eye heights and cameras. The 116 ferry photos come from one trip and one camera.
