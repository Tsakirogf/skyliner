"""§6 metrics: predictions (our label.json/meta.json per frame) vs hand labels (+ label_fixes).

Angles use each photo's focal length. 'px' in the report is the 2048-wide label grid
(1 px ~ 0.036 deg for the reference 26 mm camera), as in the requirements.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from .labels import N_COLS, load_dataset


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def score_photo(pred_dir: str, truth) -> dict:
    s = truth.scale
    row = {"id": truth.id}
    rej = os.path.join(pred_dir, "rejected.json")
    if os.path.exists(rej):
        row["rejected"] = json.load(open(rej)).get("rejected", "yes")
        return row
    pl = json.load(open(os.path.join(pred_dir, "label.json")))
    pm = json.load(open(os.path.join(pred_dir, "meta.json")))
    row["horizon_quality"] = pm.get("horizon_quality", "good")
    if row["horizon_quality"] == "none":
        row["rejected"] = "no attitude (no sea in view)"
        return row
    f = truth.meta["focal_px"]
    dpp = math.degrees(1.0 / f)               # deg per photo px
    x = truth.col_x()
    t_l0 = truth.meta_line(x)
    p_l0 = pm["horizon_a"] * x + pm["horizon_b"]
    row["horizon_err_px2048"] = float(np.median(np.abs(p_l0 - t_l0)) / s)
    row["horizon_err_deg"] = float(np.median(np.abs(p_l0 - t_l0)) * dpp)
    row["roll_err_deg"] = abs(math.degrees(math.atan(pm["horizon_a"]) - math.atan(truth.meta["horizon_a"])))
    row["horizon_source"] = pm.get("horizon_source")

    t_out = truth.level("outer")
    ign = truth.ignore_cols() | truth.occluded_cols() | truth.pseudo_cols()
    t_hor_vis = np.isfinite(truth.level("horizon"))
    t_land = np.isfinite(t_out) & ~ign
    t_sea = ~np.isfinite(t_out) & ~ign & t_hor_vis
    st = np.array(pl["state"])
    p_land = st == "land"
    p_out = _arr(pl["outer"]) * s
    row["n_truth_land"] = int(t_land.sum())
    row["n_truth_sea"] = int(t_sea.sum())
    row["n_pred_land"] = int(p_land.sum())
    row["n_pred_sea"] = int((st == "sea").sum())
    row["n_pred_unknown"] = int((st == "unknown").sum())
    row["land_recall"] = float((p_land & t_land).sum() / t_land.sum()) if t_land.sum() else float("nan")
    row["false_land"] = float((p_land & t_sea).sum() / t_sea.sum()) if t_sea.sum() else float("nan")
    row["false_land_n"] = int((p_land & t_sea).sum())
    row["sea_called_unknown_n"] = int(((st == "unknown") & t_sea).sum())
    both = p_land & t_land & np.isfinite(p_out)
    # silhouette height above the sea line
    h_err = ((p_l0 - p_out) - (t_l0 - t_out))[both] * dpp
    # absolute L1 row error (for sigma calibration: excludes horizon error)
    r_err = (p_out - t_out)[both]
    row["_h_err_deg"] = h_err.tolist()
    row["_r_err_px"] = r_err.tolist()
    row["_h_err_px"] = ((p_l0 - p_out) - (t_l0 - t_out))[both].tolist()
    row["_raw"] = _arr(pl.get("outer_raw", [None] * N_COLS))[both].tolist()
    row["_lb"] = np.array(pl.get("lower_bound", [False] * N_COLS))[both].tolist()
    row["_sig_deg"] = _arr(pl.get("sigma_deg", [None] * N_COLS))[both].tolist()
    if len(h_err):
        row["height_err_med_deg"] = float(np.median(np.abs(h_err)))
        row["height_err_p90_deg"] = float(np.percentile(np.abs(h_err), 90))
    # waterline: only checkable under distant land, where the hand-label horizon is the land/sea line
    if "waterline" in pl:
        t_hl = truth.level("horizon")
        chk = np.isfinite(t_out) & np.isfinite(t_hl) & (np.abs(t_hl - t_l0) < 3.0) & ~ign
        p_wl = _arr(pl["waterline"]) * s
        both_wl = chk & np.isfinite(p_wl)
        row["_wl_err_px2048"] = (np.abs(p_wl - t_hl)[both_wl] / s).tolist()
        row["waterline_recall"] = float(both_wl.sum() / chk.sum()) if chk.sum() else float("nan")
    # wrong ridge layer: we traced a labelled inner ridge instead of the outer one
    inn = np.fmin(np.abs(p_out - truth.level("inner1")), np.abs(p_out - truth.level("inner2")))
    wrong = both & (inn < 2.0) & (np.abs(p_out - t_out) > 3.0)
    row["wrong_layer_n"] = int(wrong.sum())
    row["n_both_land"] = int(both.sum())
    return row


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "rejected" not in r]
    he = np.concatenate([r["_h_err_deg"] for r in ok]) if ok else np.array([])
    sig = np.concatenate([r["_sig_deg"] for r in ok]) if ok else np.array([])
    tl = sum(r["n_truth_land"] for r in ok)
    rec = sum(r["land_recall"] * r["n_truth_land"] for r in ok if r["n_truth_land"]) / max(tl, 1)
    fl_rows = [r for r in ok if r["n_truth_sea"] >= 10]
    fl = sum(r["false_land_n"] for r in fl_rows) / max(sum(r["n_truth_sea"] for r in fl_rows), 1)
    cov1 = float(np.mean(np.abs(he) <= sig)) if len(he) else float("nan")
    cov2 = float(np.mean(np.abs(he) <= 2 * sig)) if len(he) else float("nan")
    return {
        "photos": len(rows), "rejected": len(rows) - len(ok),
        "horizon_low": sum(1 for r in ok if r.get("horizon_quality") == "low"),
        "horizon_err_px2048_median": float(np.median([r["horizon_err_px2048"] for r in ok])),
        "horizon_err_px2048_p90": float(np.percentile([r["horizon_err_px2048"] for r in ok], 90)),
        "roll_err_deg_median": float(np.median([r["roll_err_deg"] for r in ok])),
        "roll_err_deg_p90": float(np.percentile([r["roll_err_deg"] for r in ok], 90)),
        "height_err_deg_median": float(np.median(np.abs(he))) if len(he) else float("nan"),
        "height_err_deg_p90": float(np.percentile(np.abs(he), 90)) if len(he) else float("nan"),
        "land_recall": rec,
        "false_land": fl,
        "wrong_layer_frac": sum(r["wrong_layer_n"] for r in ok) / max(sum(r["n_both_land"] for r in ok), 1),
        "sigma_cov_1": cov1, "sigma_cov_2": cov2,
        "height_err_deg_p90_good_horizon": float(np.percentile(np.abs(np.concatenate(
            [r["_h_err_deg"] for r in ok if r.get("horizon_quality") == "good"] or [np.array([np.nan])])), 90)),
        "waterline_err_px2048_median": float(np.median(np.concatenate([r.get("_wl_err_px2048", []) for r in ok]))) if ok else float("nan"),
        "waterline_recall": float(np.nanmean([r.get("waterline_recall", np.nan) for r in ok])) if ok else float("nan"),
    }


TARGETS = [
    ("horizon_err_px2048_median", "<= 3 px", lambda v: v <= 3),
    ("roll_err_deg_median", "<= 0.05 deg", lambda v: v <= 0.05),
    ("height_err_deg_median", "<= 0.03 deg", lambda v: v <= 0.03),
    ("height_err_deg_p90", "<= 0.06 deg", lambda v: v <= 0.06),
    ("land_recall", ">= 0.90", lambda v: v >= 0.90),
    ("false_land", "<= 0.05", lambda v: v <= 0.05),
    ("wrong_layer_frac", "<= 0.02", lambda v: v <= 0.02),
    ("sigma_cov_1", "~0.68", lambda v: 0.6 <= v <= 0.76),
    ("sigma_cov_2", "~0.95", lambda v: 0.9 <= v <= 0.98),
]


def score_dir(pred: str, truth_root: str, report: str | None = None, csv_path: str | None = None):
    truth = {s.id: s for s in load_dataset(truth_root)}
    rows = []
    for fid in sorted(os.listdir(pred)):
        d = os.path.join(pred, fid)
        if fid in truth and os.path.isdir(d):
            rows.append(score_photo(d, truth[fid]))
    S = summarize(rows)
    lines = ["| metric | value | requirement | met |", "|---|---|---|---|"]
    for k, req, ok in TARGETS:
        v = S[k]
        lines.append(f"| {k} | {v:.4f} | {req} | {'yes' if ok(v) else 'NO'} |")
    lines.append(f"| photos / no attitude | {S['photos']} / {S['rejected']} | as few as possible | |")
    lines.append(f"| height_err_deg_p90, good-quality horizons only | {S['height_err_deg_p90_good_horizon']:.4f} | <= 0.06 deg | {'yes' if S['height_err_deg_p90_good_horizon'] <= 0.06 else 'NO'} |")
    lines.append(f"| waterline vs hand line under distant land (median px / recall) | {S['waterline_err_px2048_median']:.2f} / {S['waterline_recall']:.2f} | no requirement (L3 PROPOSED) | |")
    lines.append(f"| low-quality horizon (annotated, sigma inflated) | {S['horizon_low']} | | |")
    table = "\n".join(lines)
    print(table)
    if csv_path:
        keys = [k for k in rows[0] if not k.startswith("_")] if rows else []
        allk = sorted({k for r in rows for k in r if not k.startswith("_")}, key=lambda k: (k != "id", k))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=allk)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")})
    if report:
        with open(report, "w") as f:
            f.write(table + "\n")
    return S, rows
