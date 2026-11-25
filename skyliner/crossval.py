"""Out-of-fold evaluation + sigma calibration.

  python -m skyliner.crossval --data example_dataset --runs runs --folds 4 --work cv --models models

For each fold k: export fold-k weights, annotate the held-out photos with them. Then fit the
sigma calibration (models/calibration.json) on those out-of-fold residuals, re-annotate with it
and write the §6 report.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import types

import numpy as np

from .__main__ import cmd_annotate
from .export import export
from .labels import load_dataset
from .score import TARGETS, score_dir
from .train import fold_split


def run_oof(a, calib):
    samples = load_dataset(a.data)
    out = os.path.join(a.work, "oof")
    shutil.rmtree(out, ignore_errors=True)
    for k in range(a.folds):
        mdir = os.path.join(a.work, f"models_{k}")
        if not os.path.exists(os.path.join(mdir, "scene.onnx")):
            export(a.runs, str(k), mdir)
            for n in ("scene", "edge"):
                os.replace(os.path.join(mdir, f"{n}_{k}.onnx"), os.path.join(mdir, f"{n}.onnx"))
        if calib:
            json.dump(calib, open(os.path.join(mdir, "calibration.json"), "w"), indent=1)
        elif os.path.exists(os.path.join(mdir, "calibration.json")):
            os.remove(os.path.join(mdir, "calibration.json"))
        _, va = fold_split(samples, str(k), a.folds)
        sub = os.path.join(a.work, f"val_{k}")
        shutil.rmtree(sub, ignore_errors=True)
        os.makedirs(sub)
        for s in va:
            os.symlink(os.path.abspath(s.dir), os.path.join(sub, s.id))
        args = types.SimpleNamespace(input=sub, out=out, models=mdir, platform="ferry", sensor="visible:iPhone 13",
                                     eye_height=2.0, eye_height_sigma=None, focal_px=None, utc_offset=0.0,
                                     with_truth=False, copy_photo=False, no_overlay=False, threads=a.threads,
                                     strict_horizon=a.strict_horizon, emit_low_quality=False)
        cmd_annotate(args)
    return out


def _fit_bins(raw, err, lb, nbins):
    base = ~lb
    edges = np.quantile(raw[base], np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    q = []
    for i in range(nbins):
        m = base & (raw >= edges[i]) & (raw < edges[i + 1])
        q.append(float(np.quantile(err[m], 0.68)) if m.sum() > 20 else float(np.quantile(err[base], 0.68)))
    q = np.maximum.accumulate(np.maximum(q, 0.3))  # monotone in the raw spread, floor 0.3 px
    k = np.clip(np.searchsorted(edges, raw) - 1, 0, nbins - 1)
    lbf = float(np.quantile(err[lb] / q[k[lb]], 0.68)) if lb.sum() > 20 else 2.0
    return {"edges": [-1e9] + [float(e) for e in edges[1:-1]], "q68_px": [float(x) for x in q],
            "lower_bound_factor": max(1.0, lbf), "n": int(len(err))}


def fit_calibration(rows, nbins=6):
    """sigma (px) = q68 of |height-above-sea-line error| per raw-spread bin, per horizon source.
    The height error includes the horizon (pitch/roll) error, which is what Horizon sees."""
    out = {"land": {}}
    for src in ("sea", "waterline"):
        rs = [r for r in rows if r.get("horizon_source") == src and "_raw" in r and r.get("horizon_quality") == "good"]
        if not rs:
            continue
        raw = np.concatenate([r["_raw"] for r in rs])
        err = np.abs(np.concatenate([r["_h_err_px"] for r in rs]))
        lb = np.concatenate([r["_lb"] for r in rs]).astype(bool)
        ok = np.isfinite(raw) & np.isfinite(err)
        out["land"][src] = _fit_bins(raw[ok], err[ok], lb[ok], nbins)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--work", default="cv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--strict-horizon", action="store_true")
    a = ap.parse_args()
    out = run_oof(a, None)
    S0, rows = score_dir(out, a.data)
    calib = fit_calibration([r for r in rows if "rejected" not in r])
    print("calibration", json.dumps(calib))
    out = run_oof(a, calib)
    rep = os.path.join(a.work, "report.md")
    S, rows = score_dir(out, a.data, report=rep, csv_path=os.path.join(a.work, "per_photo.csv"))
    json.dump(calib, open(os.path.join(a.models, "calibration.json"), "w"), indent=1)
    json.dump({"before_calibration": S0, "after_calibration": S}, open(os.path.join(a.work, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
