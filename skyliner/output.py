"""Writers: observation.json (contract v2 JSON + PROPOSED §8 fields), label.json (hand-label
format + extra detail), meta.json, overlay.jpg.

JSON note: sigma_deg uses Infinity for unknown samples and layers use NaN for 'absent', as the
contract defines (+inf / NaN). Python's json module reads and writes these tokens.
"""
from __future__ import annotations

import calendar
import json
import math
import time

import cv2
import numpy as np

from . import __version__
from .camera import Camera
from .infer import GRID_W, LAND_ST, N_COLS, SEA_ST, STEP, UNK_ST, Annotation

SCHEMA = "horizon.observation/2+json"


def exif_time_to_ns(t: str | None, utc_offset_h: float = 0.0) -> int:
    if not t:
        return 0
    try:
        tm = time.strptime(t.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return 0
    return int((calendar.timegm(tm) - utc_offset_h * 3600) * 1_000_000_000)


def _r(x, nd=4):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return x
    return round(x, nd)


def camera_of(A: Annotation, eye_height_m: float) -> Camera:
    return Camera(A.width, A.height, A.focal_px, A.horizon_a, A.horizon_b, eye_height_m)


def build_observation(A: Annotation, *, platform_id: str, sensor_id: str, t_ns: int, eye_height_m: float,
                      eye_height_sigma_m: float | None, source: str, gnss_truth: dict | None = None) -> dict:
    cam = camera_of(A, eye_height_m)
    dpp = cam.deg_per_px()
    u = A.col_u
    st = A.state
    v = np.where(np.array(st) == LAND_ST, A.L1, A.L0_row)
    v = np.where(np.isfinite(v), v, A.L0_row)
    az, el = cam.pixel_to_angles(u, v)
    # pitch/roll uncertainty from the horizon fit adds to every sample
    if A.horizon_quality == "none":
        return None      # no sea in view: no attitude, angles cannot be computed
    low = A.horizon_quality == "low"
    # good fit: uncertainty of the fitted line; low: the line itself may be off by horizon_sigma_px
    line_sig = A.horizon_sigma_px if low else A.horizon_sigma_px * 2.0 / math.sqrt(max(A.horizon_n, 1))
    if A.horizon_source == "waterline":
        line_sig = max(line_sig, 3.0)
    sig = np.full(N_COLS, np.inf)
    line_col = A.line_sigma_px if (low and A.line_sigma_px is not None) else np.full(N_COLS, line_sig)
    for i in range(N_COLS):
        line_sig = line_col[i]
        if st[i] == LAND_ST and np.isfinite(A.L1_sigma_px[i]):
            # calibrated sigma already covers the horizon error (fitted on height-above-line errors)
            sig[i] = (A.L1_sigma_px[i] if getattr(A, "sigma_includes_horizon", False) and not low
                      else math.hypot(A.L1_sigma_px[i], line_sig)) * dpp
        elif st[i] == SEA_ST:
            sig[i] = math.hypot(max(A.horizon_sigma_px, 0.5), line_sig) * dpp
    A.sigma_deg = sig
    order = np.argsort(az)
    assert np.all(np.diff(az[order]) > 0), "azimuths must be strictly increasing"

    def layer(rows, sig_px, kind):
        _, e = cam.pixel_to_angles(u, np.where(np.isfinite(rows), rows, A.L0_row))
        e = np.where(np.isfinite(rows), e, np.nan)
        s = np.where(np.isfinite(rows), np.hypot(sig_px, line_col) * dpp, np.nan)
        return {"kind": kind, "elev_deg": [_r(x) for x in e[order]], "sigma_deg": [_r(x, 5) for x in s[order]]}

    obs = {
        "schema": SCHEMA,
        "id": A.id,
        "platform_id": platform_id,
        "sensor_id": sensor_id,
        "t_start_ns": t_ns,
        "t_end_ns": t_ns,
        "frame": "camera",
        "az_deg": [_r(x) for x in az[order]],
        "elev_deg": [_r(x) for x in el[order]],
        "state": [st[i] for i in order],
        "sigma_deg": [_r(x, 5) for x in sig[order]],
        "eye_height_m": eye_height_m,
        "roll_deg": cam.roll_deg,
        "pitch_deg": cam.pitch_deg,
        "source": source,
        # ---- PROPOSED §8 (additive; old cores ignore them)
        "layers": [layer(A.inner1, A.inner1_sigma_px, "inner_ridge"), layer(A.inner2, A.inner2_sigma_px, "inner_ridge"),
                   layer(A.waterline, A.waterline_sigma_px, "waterline")],
        "lower_bound": [bool(A.lower_bound[i]) for i in order],
        "visibility_km": float("nan"),
    }
    if eye_height_sigma_m is not None:
        obs["eye_height_sigma_m"] = eye_height_sigma_m
    if gnss_truth:
        obs["gnss_truth"] = gnss_truth
    return obs


def build_label(A: Annotation, source: str) -> dict:
    """Hand-label-compatible (2048 grid, step 4) plus per-column detail."""
    s = A.width / GRID_W
    f = lambda arr: [None if not np.isfinite(x) else round(float(x) / s, 1) for x in arr]
    return {
        "step": STEP,
        "horizon": f(A.L0_row),
        "outer": f(np.where(np.array(A.state) == LAND_ST, A.L1, np.nan)),
        "inner1": f(A.inner1),
        "inner2": f(A.inner2),
        "occl": [int(x) for x in A.occl],
        "waterline": f(A.waterline),
        "waterline_sigma": [None if not np.isfinite(x) else round(float(x) / s, 2) for x in A.waterline_sigma_px],
        "occl_type": list(A.occl_type),
        "horizon_quality": A.horizon_quality,
        # ---- extra detail (not in the hand labels)
        "state": list(A.state),
        "reason": list(A.reasons),
        "outer_sigma": [None if not np.isfinite(x) else round(float(x) / s, 2) for x in A.L1_sigma_px],
        "outer_raw": [None if not np.isfinite(x) else round(float(x), 3) for x in A.L1_raw_sigma],
        "sigma_deg": [None if not np.isfinite(x) else round(float(x), 5) for x in getattr(A, "sigma_deg", np.full(N_COLS, np.inf))],
        "lower_bound": [bool(x) for x in A.lower_bound],
        "horizon_seen": f(A.L0_obs),
        "contrast": [None if not np.isfinite(x) else round(float(x), 3) for x in A.contrast],
        "source": source,
    }


def build_meta(A: Annotation, name: str, time_str: str | None, eye_height_m: float) -> dict:
    cam = camera_of(A, eye_height_m)
    return {
        "width": A.width, "height": A.height, "focal_px": A.focal_px,
        "horizon_a": A.horizon_a, "horizon_b": A.horizon_b,
        "name": name, "time": time_str,
        "roll_deg": cam.roll_deg, "pitch_deg": cam.pitch_deg,
        "horizon_source": A.horizon_source, "horizon_quality": A.horizon_quality, "horizon_note": A.horizon_note,
        "horizon_sigma_px": A.horizon_sigma_px, "horizon_points": A.horizon_n,
        "attitude": A.horizon_quality != "none",
        "obstacles": {t: sum(1 for o in A.obstacles if o["type"] == t) for t in sorted({o["type"] for o in A.obstacles})},
        "waterline_columns": int(np.isfinite(A.waterline).sum()),
        "eye_height_m": eye_height_m,
        "counts": {k: int(sum(1 for x in A.state if x == k)) for k in (LAND_ST, SEA_ST, UNK_ST)},
        "timings_s": {k: round(v, 3) for k, v in A.timings.items()},
    }


def overlay(img_rgb: np.ndarray, A: Annotation) -> np.ndarray:
    im = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    u = A.col_u
    th = max(1, A.width // 800)

    def poly(rows, color, mask=None, maxjump=None):
        for i in range(N_COLS - 1):
            ok = np.isfinite(rows[i]) and np.isfinite(rows[i + 1])
            if ok and maxjump is not None and abs(rows[i + 1] - rows[i]) > maxjump:
                ok = False
            if mask is not None:
                ok = ok and mask[i] and mask[i + 1]
            if ok:
                cv2.line(im, (int(u[i]), int(round(rows[i]))), (int(u[i + 1]), int(round(rows[i + 1]))), color, th, cv2.LINE_AA)

    # obstacles: translucent magenta, with their type
    if A.p_occ is not None:
        m = cv2.resize(A.p_occ.astype(np.float32), (A.width, A.height)) > 0.5
        im[m] = (0.55 * im[m] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
        for o in A.obstacles:
            x0, y0, x1, y1 = [int(v) for v in o["bbox_px"]]
            cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), max(1, th))
            cv2.putText(im, o["type"], (x0 + 4, max(y0 + 28, 40)), 0, 0.9 * A.width / 1600, (255, 255, 255), 2)
    if np.isfinite(A.horizon_a) and A.horizon_quality != "none":
        poly(A.L0_row, (255, 255, 0) if A.horizon_quality == "good" else (0, 255, 255))
    poly(A.waterline, (0, 220, 255), maxjump=6)
    land = np.array(A.state) == LAND_ST
    poly(A.L1, (0, 0, 255), land & ~A.lower_bound)
    poly(A.L1, (0, 165, 255), land & A.lower_bound)
    poly(A.inner1, (0, 255, 0), maxjump=4)
    poly(A.inner2, (255, 0, 255), maxjump=4)
    bar = max(6, A.height // 100)
    colors = {LAND_ST: (0, 0, 200), SEA_ST: (200, 120, 0), UNK_ST: (128, 128, 128)}
    for i in range(N_COLS):
        x0 = int(u[i]); x1 = int(u[i + 1]) if i + 1 < N_COLS else A.width
        c = (255, 0, 255) if A.occl[i] else colors[A.state[i]]
        cv2.rectangle(im, (x0, 0), (x1, bar), c, -1)
    if A.horizon_quality in ("low", "none"):
        msg = {"low": f"horizon: LOW quality (sigma {A.horizon_sigma_px:.1f} px, {A.horizon_source})",
               "none": "horizon: NONE (no sea in view) - pixel annotation only, no observation"}[A.horizon_quality]
        cv2.putText(im, msg, (20, 3 * bar + 30), 0, 1.0 * A.width / 1600, (0, 255, 255), 2)
    if A.rejected:
        cv2.putText(im, "REJECTED: " + A.rejected, (20, 3 * bar + 30), 0, 1.0, (0, 0, 255), 2)
    return im


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, allow_nan=True)


def source_string(model_info: dict) -> str:
    sc = model_info.get("scene", {}).get("sha256_10", "?")
    ed = model_info.get("edge", {}).get("sha256_10", "?")
    return f"skyliner {__version__} (scene {sc}, edge {ed})"
