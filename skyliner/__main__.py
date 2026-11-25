"""Skyliner CLI.

  python -m skyliner annotate INPUT OUTDIR [--models models] [--platform ferry] [--eye-height 2]
         INPUT = a folder of frames (each a subfolder with photo.jpg [+ meta.json] [+ gps.json]),
                 a folder of .jpg files, or a single .jpg
  python -m skyliner score PREDDIR TRUTHDIR [--report report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

from .output import (build_label, build_meta, build_observation, dump_json, exif_time_to_ns, overlay,
                     source_string)
from .obstacles import TYPES as OBSTACLE_TYPES, full_res_mask
from .sensors import focal_from_exif


def iter_inputs(path):
    """Yield (id, photo_path, meta_or_None, gps_or_None)."""
    def j(p):
        try:
            return json.load(open(p)) if os.path.exists(p) else None
        except json.JSONDecodeError:
            return None
    if os.path.isfile(path):
        yield os.path.splitext(os.path.basename(path))[0], path, None, None
        return
    for n in sorted(os.listdir(path)):
        p = os.path.join(path, n)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "photo.jpg")):
            yield n, os.path.join(p, "photo.jpg"), j(os.path.join(p, "meta.json")), j(os.path.join(p, "gps.json"))
        elif n.lower().endswith((".jpg", ".jpeg")):
            yield os.path.splitext(n)[0], p, None, None


def cmd_annotate(a):
    from .infer import Annotator
    ann = Annotator(a.models, threads=a.threads, waterline_fallback=not getattr(a, 'strict_horizon', False))
    src = source_string(ann.model_info)
    os.makedirs(a.out, exist_ok=True)
    n_ok = n_rej = 0
    rows = []
    for fid, photo, meta, gps in iter_inputs(a.input):
        bgr = cv2.imread(photo, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"{fid}: cannot read image", file=sys.stderr)
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        focal, t_str, sensor = None, None, a.sensor
        if meta:
            focal = meta.get("focal_px")
            if meta.get("width") and meta["width"] != W:
                focal = focal * W / meta["width"]
            t_str = meta.get("time")
        if focal is None or a.focal_px:
            ex = focal_from_exif(photo, W, H)
            focal = a.focal_px or (ex and ex["focal_px"])
            t_str = t_str or (ex and ex.get("time"))
            sensor = (ex and ex.get("sensor_id")) or sensor
        if not focal:
            print(f"{fid}: REJECTED no focal length (no meta.json, no EXIF, no --focal-px)", file=sys.stderr)
            n_rej += 1
            continue
        A = ann.annotate(img, float(focal), id=fid, eye_height_m=a.eye_height)
        od = os.path.join(a.out, fid)
        os.makedirs(od, exist_ok=True)
        if A.rejected:
            n_rej += 1
            dump_json({"id": fid, "rejected": A.rejected, "source": src}, os.path.join(od, "rejected.json"))
            print(f"{fid}: REJECTED {A.rejected}")
            continue
        t_ns = exif_time_to_ns(t_str, a.utc_offset)
        truth = None
        if gps and a.with_truth:
            truth = {k: gps[k] for k in ("lat", "lon", "alt_m", "error_m") if k in gps}
        obs = build_observation(A, platform_id=a.platform, sensor_id=sensor, t_ns=t_ns, eye_height_m=a.eye_height,
                                eye_height_sigma_m=a.eye_height_sigma, source=src, gnss_truth=truth)
        if obs is not None and A.horizon_quality == "low" and not a.emit_low_quality:
            # full annotation is written; the observation is kept out of Horizon's way by default
            # (Horizon weighs sigma only inside a bin today, so an uncertain horizon can mislead it)
            dump_json(obs, os.path.join(od, "observation.low_quality.json"))
        elif obs is not None:
            dump_json(obs, os.path.join(od, "observation.json"))
        else:  # no sea in view: no attitude -> nothing for Horizon; pixel annotation still written
            p_obs = os.path.join(od, "observation.json")
            if os.path.exists(p_obs):
                os.replace(p_obs, os.path.join(od, "observation.stale.json"))
        dump_json({"obstacles": A.obstacles, "types": OBSTACLE_TYPES,
                   "mask": "obstacle_mask.png (photo resolution, 255 = obstacle)",
                   "note": "type is a heuristic on geometry and colour; the network segments one obstacle class"},
                  os.path.join(od, "obstacles.json"))
        cv2.imwrite(os.path.join(od, "obstacle_mask.png"), full_res_mask(A.p_occ, W, H) * 255)
        dump_json(build_label(A, src), os.path.join(od, "label.json"))
        dump_json(build_meta(A, (meta or {}).get("name") or os.path.basename(photo), t_str, a.eye_height),
                  os.path.join(od, "meta.json"))
        if gps:
            dump_json(gps, os.path.join(od, "gps.json"))
        if a.copy_photo:
            shutil.copy(photo, os.path.join(od, "photo.jpg"))
        if not a.no_overlay:
            cv2.imwrite(os.path.join(od, "overlay.jpg"), overlay(img, A), [cv2.IMWRITE_JPEG_QUALITY, 85])
        n_ok += 1
        c = {k: sum(1 for x in A.state if x == k) for k in ("land", "sea", "unknown")}
        att = (f"roll {obs['roll_deg']:+.3f} pitch {obs['pitch_deg']:+.3f}" if obs else "no attitude")
        print(f"{fid}: land {c['land']:3d} sea {c['sea']:3d} unknown {c['unknown']:3d}  horizon {A.horizon_quality:4s} "
              f"{A.horizon_source:9s} obstacles {len(A.obstacles)} waterline {int(np.isfinite(A.waterline).sum()):3d}  "
              f"{att}  {A.timings.get('total_s', 0):.2f}s", flush=True)
    print(f"done: {n_ok} annotated, {n_rej} rejected -> {a.out}")


def cmd_score(a):
    from .score import score_dir
    score_dir(a.pred, a.truth, a.report, a.csv)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skyliner")
    sp = ap.add_subparsers(dest="cmd", required=True)
    an = sp.add_parser("annotate", help="annotate photos -> observation/label/meta per frame")
    an.add_argument("input")
    an.add_argument("out")
    an.add_argument("--models", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    an.add_argument("--platform", default="ferry")
    an.add_argument("--sensor", default="visible:iPhone 13")
    an.add_argument("--eye-height", type=float, default=2.0, help="camera height above sea (m); 2 m = database eye")
    an.add_argument("--eye-height-sigma", type=float, default=None)
    an.add_argument("--focal-px", type=float, default=None, help="override focal length (photo pixels)")
    an.add_argument("--utc-offset", type=float, default=0.0, help="hours to subtract from the photo time to get UTC")
    an.add_argument("--with-truth", action="store_true", help="copy gps.json into observation.gnss_truth (evaluation only)")
    an.add_argument("--copy-photo", action="store_true")
    an.add_argument("--no-overlay", action="store_true")
    an.add_argument("--threads", type=int, default=0)
    an.add_argument("--emit-low-quality", action="store_true",
                    help="write observation.json also for low-quality horizons (default: observation.low_quality.json)")
    an.add_argument("--strict-horizon", action="store_true",
                    help="reject frames without an open-sea horizon (requirements §4.2) instead of using the land/sea waterline as L0")
    an.set_defaults(fn=cmd_annotate)
    sc = sp.add_parser("score", help="§6 metrics of predictions against hand labels")
    sc.add_argument("pred")
    sc.add_argument("truth")
    sc.add_argument("--report", default=None)
    sc.add_argument("--csv", default=None)
    sc.set_defaults(fn=cmd_score)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
