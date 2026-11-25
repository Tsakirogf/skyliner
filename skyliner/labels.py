"""Hand-label I/O and dense training targets.

Hand-label format (data/skyline_labels, example_dataset/*/label.json):
  step=4 on a 2048-px-wide grid -> 512 columns; rows in the same 2048 grid.
  horizon (sea line), outer (L1), inner1/inner2 (L2), occl (unused so far). None = no land.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

LABEL_GRID_W = 2048
N_COLS = 512
STEP = 4

SKY, LAND, SEA, OCCL, IGNORE = 0, 1, 2, 3, 255
CLASS_NAMES = ["sky", "land", "sea", "occluder"]

_FIX_PATH = os.path.join(os.path.dirname(__file__), "label_fixes.json")


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def load_fixes(path: str = _FIX_PATH) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


@dataclass
class Sample:
    id: str
    dir: str
    meta: dict
    label: dict | None
    gps: dict | None
    obs: dict | None
    fixes: dict = field(default_factory=dict)

    @property
    def W(self):
        return int(self.meta["width"])

    @property
    def H(self):
        return int(self.meta["height"])

    @property
    def scale(self):  # label grid -> photo pixels
        return self.W / LABEL_GRID_W

    def image(self):
        im = cv2.imread(os.path.join(self.dir, "photo.jpg"), cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(f"cannot read {self.dir}/photo.jpg")
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # --- label columns (photo pixel units) ---
    def col_x(self):
        return np.arange(N_COLS) * STEP * self.scale

    def level(self, name):
        v = _arr(self.label[name])
        if name == "outer" and self.fixes.get("outer_pseudo"):
            ps = _arr(self.fixes["outer_pseudo"])
            v = np.where(np.isfinite(v), v, ps)
        return v * self.scale

    def pseudo_cols(self):
        """Label columns filled from a pseudo-label (trained on, never scored)."""
        if not self.fixes.get("outer_pseudo"):
            return np.zeros(N_COLS, bool)
        return np.isfinite(_arr(self.fixes["outer_pseudo"])) & ~np.isfinite(_arr(self.label["outer"]))

    def meta_line(self, x):
        return self.meta["horizon_a"] * np.asarray(x, float) + self.meta["horizon_b"]

    def occluder_mask(self, shape=None):
        H, W = shape if shape else (self.H, self.W)
        m = np.zeros((H, W), np.uint8)
        for poly in self.fixes.get("occluders", []):
            p = np.array(poly, np.float64)
            p[:, 0] *= W / self.W
            p[:, 1] *= H / self.H
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
        return m

    def ignore_cols(self):
        """Boolean per label column: annotation of land above the horizon not trusted."""
        ig = np.zeros(N_COLS, bool)
        for a, b in self.fixes.get("ignore_cols", []):
            ig[a:b + 1] = True
        ig |= spike_cols(_arr(self.label["outer"]))
        return ig

    def occluded_cols(self, band_px=6):
        """Label columns where an occluder polygon touches the L0/L1 band."""
        if not self.fixes.get("occluders"):
            return np.zeros(N_COLS, bool)
        m = self.occluder_mask()
        x = np.clip(np.round(self.col_x()).astype(int), 0, self.W - 1)
        hor, out = self.meta_line(self.col_x()), self.level("outer")
        res = np.zeros(N_COLS, bool)
        for i in range(N_COLS):
            top = out[i] if np.isfinite(out[i]) else hor[i]
            if not np.isfinite(hor[i]):
                continue
            y0 = int(max(0, np.floor(min(top, hor[i]) - band_px)))
            y1 = int(min(self.H, np.ceil(max(top, hor[i]) + band_px)))
            res[i] = m[y0:y1, x[i]].any() or m[:y0, x[i]].any()
        return res


def spike_cols(outer, thr=10.0):
    """Columns where the label draws a thin vertical spike (mast, ship): narrow excursions."""
    o = outer.copy()
    res = np.zeros(len(o), bool)
    fin = np.isfinite(o)
    for i in range(len(o)):
        if not fin[i]:
            continue
        lo, hi = max(0, i - 4), min(len(o), i + 5)
        nb = o[lo:hi][np.isfinite(o[lo:hi])]
        if len(nb) >= 5 and abs(o[i] - np.median(nb)) > thr:
            res[i] = True
    # widen by one column
    res = res | np.roll(res, 1) | np.roll(res, -1)
    return res


def load_sample(d: str, fixes: dict | None = None) -> Sample:
    def j(n):
        p = os.path.join(d, n)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    sid = os.path.basename(os.path.normpath(d))
    fixes = load_fixes() if fixes is None else fixes
    return Sample(sid, d, j("meta.json"), j("label.json"), j("gps.json"), j("observation.json"), fixes.get(sid, {}))


def load_dataset(root: str) -> list[Sample]:
    fixes = load_fixes()
    out = []
    for n in sorted(os.listdir(root)):
        d = os.path.join(root, n)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "photo.jpg")):
            out.append(load_sample(d, fixes))
    return out


# ---------------------------------------------------------------- dense targets
def to_pixel_columns(rows_labelcols: np.ndarray, W: int, scale: float) -> np.ndarray:
    """Interpolate a per-label-column row array onto every pixel column (NaN-aware)."""
    t = np.arange(W) / (STEP * scale)
    i0 = np.clip(np.floor(t).astype(int), 0, N_COLS - 1)
    i1 = np.clip(i0 + 1, 0, N_COLS - 1)
    w = t - i0
    r0, r1 = rows_labelcols[i0], rows_labelcols[i1]
    both = np.isfinite(r0) & np.isfinite(r1)
    out = np.where(w < 0.5, r0, r1)
    out[both] = (r0 * (1 - w) + r1 * w)[both]
    return out


@dataclass
class Targets:
    cls: np.ndarray          # HxW uint8 (SKY/LAND/SEA/OCCL/IGNORE)
    rows: dict               # name -> per pixel column row (NaN = absent): L0vis, L1, L2a, L2b, L0
    edge_valid: np.ndarray   # per pixel column bool: heatmap supervision trusted


def build_targets(s: Sample) -> Targets:
    W, H, sc = s.W, s.H, s.scale
    # L0: the fitted sea line from meta (labels set horizon=None where it is hidden, and
    # occasionally follow a near waterline). Visible-L0 columns: label present and on the line.
    hor_lab = to_pixel_columns(s.level("horizon"), W, sc)
    hor = s.meta["horizon_a"] * np.arange(W) + s.meta["horizon_b"]
    hor_visible = np.isfinite(hor_lab) & (np.abs(hor_lab - hor) < 3.0)
    out = to_pixel_columns(s.level("outer"), W, sc)
    in1 = to_pixel_columns(s.level("inner1"), W, sc)
    in2 = to_pixel_columns(s.level("inner2"), W, sc)
    ig_lab = s.ignore_cols()
    ig = to_pixel_columns(ig_lab.astype(float), W, sc) > 0.25
    occ = s.occluder_mask()

    yy = np.arange(H)[:, None].astype(np.float32)
    top = np.where(np.isfinite(out), out, hor)[None, :]
    cls = np.full((H, W), SKY, np.uint8)
    cls[(yy >= top) & (yy < hor[None, :])] = LAND
    cls[yy >= hor[None, :]] = SEA
    cls[:, ~np.isfinite(hor)] = IGNORE
    # untrusted land annotation: ignore everything above the sea line (+ a margin)
    cls[(yy < hor[None, :] + 3) & ig[None, :]] = IGNORE
    cls[occ > 0] = OCCL

    occ_cols = (occ[: int(np.nanmax(hor)) + 10].sum(0) > 0) if np.isfinite(hor).any() else np.zeros(W, bool)
    # occluder above or within the band => that column's edges unsupervised
    edge_valid = np.isfinite(hor) & ~ig & ~occ_cols
    l0vis = np.where(np.isfinite(out) | ~hor_visible, np.nan, hor)
    # L0 under land is the (invisible) extrapolated line, not an image edge
    # land->sea line under distant land: what the labels (and Horizon) use as L0 when no open
    # sea is visible. Only where the label horizon is given there and sits on the fitted line.
    l0land = np.where(np.isfinite(out) & hor_visible, hor, np.nan)
    rows = dict(L0=hor, L0vis=l0vis, L1=out, L2a=in1, L2b=in2, L0land=l0land)
    return Targets(cls, rows, edge_valid)


def heatmaps(rows: dict, y0: int, y1: int, x0: int, x1: int, sigma: float = 1.0) -> np.ndarray:
    """4 x (y1-y0) x (x1-x0) gaussian row heatmaps: [L0vis, L1, L2(a|b), L0land] for a crop."""
    yy = np.arange(y0, y1, dtype=np.float32)[:, None]

    def g(r):
        r = r[x0:x1][None, :]
        h = np.exp(-0.5 * ((yy - r) / sigma) ** 2)
        return np.where(np.isfinite(r), h, 0.0)

    l2 = np.maximum(g(rows["L2a"]), g(rows["L2b"]))
    return np.stack([g(rows["L0vis"]), g(rows["L1"]), l2, g(rows["L0land"])]).astype(np.float32)
