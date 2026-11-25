"""Inference: photo -> per-column skyline annotation (all levels), using the ONNX models.

Runtime deps: numpy, opencv, onnxruntime. No torch.

Pipeline
  1. SceneNet (512x384)       -> P(sky, land, sea, occluder) per pixel
  2. coarse L0 candidates      -> sky->sea transitions in columns with no land
  3. EdgeNet (full res, band)  -> heatmaps L0 / L1 / L2
  4. L0 refine + robust line   -> horizon_a, horizon_b (roll, pitch)
  5. per label column (512 cols, 4 px at 2048 width):
       L1 topmost-wins search, state rules (SKYLINER_REQUIREMENTS §5), sigma, lower_bound,
       L2 inner ridges (up to 2, chained), occluder flag
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera
from .obstacles import instances as obstacle_instances

N_COLS, STEP, GRID_W = 512, 4, 2048
SCENE_W, SCENE_H = 512, 384
SKY, LAND, SEA, OCCL = 0, 1, 2, 3

SEA_ST, LAND_ST, UNK_ST = "sea", "land", "unknown"


def _softmax(x, axis=0):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Annotation:
    id: str
    width: int
    height: int
    focal_px: float
    rejected: str | None = None
    horizon_a: float = float("nan")
    horizon_b: float = float("nan")
    horizon_source: str = ""          # "sea" (open-sea line) | "waterline" (fallback) | ""
    horizon_sigma_px: float = float("nan")
    horizon_n: int = 0
    col_u: np.ndarray = None          # photo-pixel x of each column
    state: list = None
    L1: np.ndarray = None             # rows (photo px), NaN where no land
    L1_sigma_px: np.ndarray = None
    L1_raw_sigma: np.ndarray = None   # uncalibrated spread feature
    L0_row: np.ndarray = None         # sea-line row per column (line)
    L0_obs: np.ndarray = None         # refined visible L0 per column (NaN if not seen)
    inner1: np.ndarray = None
    inner2: np.ndarray = None
    inner1_sigma_px: np.ndarray = None
    inner2_sigma_px: np.ndarray = None
    occl: np.ndarray = None           # bool per column
    lower_bound: np.ndarray = None    # bool per column
    contrast: np.ndarray = None       # land/sky contrast at L1 (0..1)
    reasons: list = None              # per-column short reason for unknown
    horizon_quality: str = ""         # good | low (weak fit: sigma inflated) | none (no sea at all)
    waterline: np.ndarray = None      # L3: where land meets the sea, rows (NaN = absent)
    waterline_sigma_px: np.ndarray = None
    occl_type: list = None            # per column: type of the obstacle hiding the skyline ("" = none)
    obstacles: list = None            # obstacle instances (see obstacles.py)
    obstacle_ids: np.ndarray = None   # scene-resolution instance id image
    p_occ: np.ndarray = None          # scene-resolution obstacle probability
    line_sigma_px: np.ndarray = None  # per-column sea-line uncertainty when horizon_quality == low
    horizon_note: str = ""
    timings: dict = field(default_factory=dict)


class Annotator:
    def __init__(self, model_dir: str, scene_file="scene.onnx", edge_file="edge.onnx", threads: int = 0,
                 waterline_fallback: bool = True):
        import onnxruntime as ort
        self.waterline_fallback = waterline_fallback
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.scene = ort.InferenceSession(os.path.join(model_dir, scene_file), so, providers=["CPUExecutionProvider"])
        self.edge = ort.InferenceSession(os.path.join(model_dir, edge_file), so, providers=["CPUExecutionProvider"])
        cp = os.path.join(model_dir, "calibration.json")
        self.calib = json.load(open(cp)) if os.path.exists(cp) else None
        mp = os.path.join(model_dir, "models.json")
        self.model_info = json.load(open(mp)) if os.path.exists(mp) else {}

    # ------------------------------------------------------------------ nets
    def run_scene(self, img):
        x = cv2.resize(img, (SCENE_W, SCENE_H), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32).transpose(2, 0, 1)[None]
        y = self.scene.run(None, {"x": x})[0][0]
        return _softmax(y, 0)  # 4 x 384 x 512

    def run_edge(self, img, y0, y1):
        y0 = max(0, int(y0)); y1 = min(img.shape[0], int(y1))
        y1 = max(y1, y0 + 16)
        crop = img[y0:y1]
        pad = (-crop.shape[0]) % 4
        if pad:
            crop = np.pad(crop, ((0, pad), (0, 0), (0, 0)), mode="edge")
        x = crop.astype(np.float32).transpose(2, 0, 1)[None]
        h = _sigmoid(self.edge.run(None, {"x": x})[0][0])
        return h[:, : y1 - y0], y0

    # ------------------------------------------------------------------ main
    def annotate(self, img: np.ndarray, focal_px: float, id: str = "frame", eye_height_m: float = 2.0) -> Annotation:
        import time
        t0 = time.time()
        H, W = img.shape[:2]
        A = Annotation(id=id, width=W, height=H, focal_px=focal_px)
        P = self.run_scene(img)
        A.timings["scene_s"] = time.time() - t0
        sx, sy = W / SCENE_W, H / SCENE_H
        cls = P.argmax(0)

        s = W / GRID_W
        u = np.arange(N_COLS) * STEP * s
        A.col_u = u
        sc = np.clip(np.round((u + 0.5) / sx - 0.5).astype(int), 0, SCENE_W - 1)  # scene column per label col

        # ---- coarse per-column topology from the scene net
        coarse_l1 = np.full(N_COLS, np.nan)   # photo rows
        first_cls = np.full(N_COLS, -1)
        top_land = np.zeros(N_COLS, bool)
        sea_edge = np.full(N_COLS, np.nan)    # sky->sea transition rows (visible horizon)
        wl_edge = np.full(N_COLS, np.nan)     # land->sea transition rows (waterline)
        faint_gap = np.zeros(N_COLS)          # rows between P_land>0.2 and >0.5 tops (haze)
        for i, c in enumerate(sc):
            col = cls[:, c]
            nonsky = np.where(col != SKY)[0]
            if len(nonsky) == 0:
                continue
            r0 = nonsky[0]
            # require 2 consecutive rows to avoid single-pixel noise
            for r in nonsky:
                if r + 1 < SCENE_H and col[r + 1] != SKY:
                    r0 = r
                    break
            first_cls[i] = col[r0]
            top_land[i] = r0 == 0
            if col[r0] == LAND:
                # sub-row: interpolate where P_land crosses P_sky
                pl, ps = P[LAND, :, c], P[SKY, :, c]
                d = pl - ps
                rr = r0
                if rr > 0 and d[rr - 1] < 0 <= d[rr]:
                    t = d[rr - 1] / (d[rr - 1] - d[rr])
                    rr = rr - 1 + t
                coarse_l1[i] = (rr + 0.5) * sy - 0.5
                lo = np.where(pl[: r0 + 1] > 0.2)[0]
                if len(lo):
                    faint_gap[i] = (r0 - lo[0]) * sy
            # first sea row from the top and what sits right above it
            # top of the topmost long sea run (>= 8 scene rows): skips short sea-coloured patches
            # inside land, and sea seen below/between railings further down the frame
            rb = 0
            isea = (col == SEA).astype(np.int8)
            d = np.diff(np.concatenate([[0], isea, [0]]))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            for st_, en_ in zip(starts, ends):
                if en_ - st_ >= 8 and st_ > 0:
                    rb = st_
                    break
            if rb > 0:
                above = col[rb - 1]
                if above in (SKY, LAND):
                    d = P[SEA, :, c] - P[above, :, c]
                    rr = float(rb)
                    if d[rb - 1] < 0 <= d[rb]:
                        rr = rb - 1 + d[rb - 1] / (d[rb - 1] - d[rb])
                    y = (rr + 0.5) * sy - 0.5
                    if above == SKY:
                        sea_edge[i] = y
                    else:
                        wl_edge[i] = y
        # obstacles: mask out of the sea-line evidence (machinery or railing edges must not be
        # taken for the horizon)
        occm = P[OCCL] > 0.5
        sea_edge_all, wl_edge_all = sea_edge.copy(), wl_edge.copy()
        for i, c in enumerate(sc):
            for arr in (sea_edge, wl_edge):
                if np.isfinite(arr[i]):
                    r = int(arr[i] / sy)
                    if occm[max(0, r - 4): r + 5, c].any():
                        arr[i] = np.nan
        A.p_occ = P[OCCL]
        A.timings["topology_s"] = time.time() - t0

        # ---- coarse line through every sky->sea and land->sea transition (the labels end
        # distant land at the sea line, so the scene net does too)
        allt = np.where(np.isfinite(sea_edge), sea_edge, wl_edge)
        line, _ = self._fit_line(u, allt, thr=4 * sy)
        masked_out = False
        if line is None:
            # obstacles hide almost all of the sea line: use the unmasked evidence, quality low
            allt = np.where(np.isfinite(sea_edge_all), sea_edge_all, wl_edge_all)
            line, _ = self._fit_line(u, allt, thr=4 * sy)
            masked_out = line is not None
        if line is None:
            # last resort: a few sea-line points only (e.g. a strip of sea beside the machinery):
            # level line through them; the 3 deg roll term in line_sigma_px carries the doubt
            pts = np.isfinite(allt)
            if pts.sum() >= 4:
                line, masked_out = (0.0, float(np.median(allt[pts]))), True
        no_line = line is None
        if no_line:
            # no sea in view at all: annotate in pixels (silhouette, obstacles), no attitude
            a, b = 0.0, float(H - 1)
        else:
            a, b = line
        Lrow = a * u + b

        # ---- EdgeNet on the band [topmost land - margin, horizon + margin]
        tops = coarse_l1[np.isfinite(coarse_l1)]
        y_top = min(np.nanmin(Lrow), tops.min() if len(tops) else np.inf) - 3 * sy - 12
        y_bot = max(np.nanmax(Lrow) if not no_line else 0.0,
                    np.nanmax(wl_edge) if np.isfinite(wl_edge).any() else 0.0,
                    np.nanmax(sea_edge) if np.isfinite(sea_edge).any() else 0.0) + 6 * sy + 12
        t1 = time.time()
        Hm, y0 = self.run_edge(img, y_top, y_bot)
        A.timings["edge_s"] = time.time() - t1

        def heat_col(ch, uu):
            x0 = int(np.floor(uu)); w = uu - x0
            x0 = min(max(x0, 0), W - 1); x1 = min(x0 + 1, W - 1)
            return Hm[ch, :, x0] * (1 - w) + Hm[ch, :, x1] * w

        # ---- refine the sea line. Narrow search around the coarse line first (3 scene rows);
        # widen to 6 only if that finds too few points (the coarse line can sit a few rows low).
        Lc = Lrow.copy()

        def refine(k):
            open_sea = np.zeros(N_COLS, bool)
            L0obs = np.full(N_COLS, np.nan)
            # open-sea columns: EdgeNet L0 (trained on sky-over-sea only) fires on the line, and
            # any land the scene net sees there is only a sliver at the line
            for i in range(N_COLS):
                if blocked[i]:
                    continue
                thin = (not np.isfinite(coarse_l1[i])) or (Lc[i] - coarse_l1[i] <= (k - 0.5) * sy)
                if not thin:
                    continue
                r, pk, _ = _peak_near(heat_col(0, u[i]), Lc[i] - y0, k * sy)
                if r is None or pk < 0.3:
                    continue
                _, p1, _ = _peak_near(heat_col(1, u[i]), r - 2 * sy, 2.5 * sy)
                if p1 > pk:
                    continue
                open_sea[i] = True
                L0obs[i] = r + y0
            n_open = int(open_sea.sum())
            src = "sea" if n_open >= 12 else "waterline"
            if src == "waterline":
                L0obs[:] = np.nan
            # land->sea line under distant land (EdgeNet channel 3). In "sea" frames it pins the
            # slope when the open sea shows on one side only; near-shore waterlines sit below the
            # line and are rejected by the robust fit.
            near = np.isfinite(wl_edge) & (np.abs(wl_edge - Lc) < (k + 1) * sy) & ~open_sea & ~blocked
            for i in np.where(near)[0]:
                if Hm.shape[0] >= 4:
                    r, pk, _ = _peak_near(heat_col(3, u[i]), Lc[i] - y0, k * sy)
                    if r is not None and pk > (0.35 if src == "sea" else 0.25):
                        L0obs[i] = r + y0
                elif src == "waterline":
                    L0obs[i] = wl_edge[i]
            fit = _robust_line(u, L0obs)
            okp = np.isfinite(L0obs)
            span = (u[okp].max() - u[okp].min()) / W if okp.sum() >= 2 else 0.0
            good = fit is not None and fit[3] >= 24 and span >= 0.25
            if good:
                # sanity: the refined line must agree with the coarse one (no tilt jumps)
                d_ang = abs(math.degrees(math.atan(fit[0]) - math.atan((Lc[-1] - Lc[0]) / (u[-1] - u[0]))))
                d_off = np.abs((fit[0] * u + fit[1]) - Lc).max()
                # and the points must actually lie on one line (bad fits through deck machinery
                # had 8-11 px residuals; good ones <= ~3 px)
                good = d_ang < 2.0 and d_off < (k + 6) * sy and fit[2] < 5.0 * W / 1600
            return good, fit, open_sea, L0obs, src, int(okp.sum()), span

        # columns where an obstacle sits on the coarse sea line give no horizon evidence
        blocked = np.zeros(N_COLS, bool)
        for i, c in enumerate(sc):
            r = int(Lc[i] / sy)
            blocked[i] = occm[max(0, r - 6): r + 7, c].any()

        if no_line:
            good, fit, open_sea, L0obs, src, npts, span = False, None, np.zeros(N_COLS, bool), \
                np.full(N_COLS, np.nan), "none", 0, 0.0
        else:
            good, fit, open_sea, L0obs, src, npts, span = refine(3)
            if not good:
                g6 = refine(6)
                if g6[0] or (fit is None and g6[1] is not None):
                    good, fit, open_sea, L0obs, src, npts, span = g6
        if src == "waterline" and not self.waterline_fallback:
            A.rejected = "no open-sea horizon visible (no attitude reference); strict horizon mode"
            A.state = [UNK_ST] * N_COLS
            return A
        if no_line:
            quality, sig, n = "none", float("nan"), 0
        elif good and not masked_out:
            quality = "good"
            a, b, sig, n = fit
        else:
            # weak evidence (sea line mostly hidden by obstacles, near shore): keep the best line,
            # flag it, and carry its uncertainty into every sample's sigma instead of rejecting
            quality = "low"
            if masked_out and fit is None:
                fit = None
            if fit is not None and fit[3] >= 6:
                fa, fb, fs, n = fit
                disagreement = float(np.abs((fa * u + fb) - Lc).max())
                if disagreement < 12 * sy:
                    a, b = fa, fb
                    sig = max(fs, disagreement / 2, 1.5 * sy)
                else:
                    sig, n = max(disagreement / 2, 3 * sy), int(np.isfinite(allt).sum())
            else:
                sig, n = 3.0 * sy, int(np.isfinite(allt).sum())
        A.horizon_a, A.horizon_b, A.horizon_source = float(a), float(b), src
        A.horizon_sigma_px, A.horizon_n, A.horizon_quality = float(sig), int(n), quality
        if quality == "low":
            # the line may be shifted AND tilted: offset term + an assumed 3 deg roll uncertainty
            # that grows away from the evidence centre
            okp = np.isfinite(L0obs)
            uc = float(u[okp].mean()) if okp.any() else W / 2
            A.line_sigma_px = np.hypot(max(sig, 0.03 * H), np.abs(u - uc) * math.tan(math.radians(3.0)))
        else:
            A.line_sigma_px = None
        Lrow = a * u + b
        A.L0_row, A.L0_obs = Lrow, L0obs

        # ---- obstacle instances (typed), per-column lookup
        line_at = None if no_line else (lambda x: a * x + b)
        A.obstacles, A.obstacle_ids = obstacle_instances(P[OCCL], img, line_at)
        id2type = {o["id"]: o["type"] for o in A.obstacles}
        occl_type = [""] * N_COLS

        # ---- per-column L1 / state
        st = [UNK_ST] * N_COLS
        reasons = [""] * N_COLS
        L1 = np.full(N_COLS, np.nan)
        L1raw = np.full(N_COLS, np.nan)
        contrast = np.full(N_COLS, np.nan)
        lower = np.zeros(N_COLS, bool)
        occl = np.zeros(N_COLS, bool)
        cam_band = 3 * sy
        imgf = img.astype(np.float32) / 255.0
        for i in range(N_COLS):
            c = sc[i]
            colc = cls[:, c]
            # occluder touching the skyline band [top of interest - m, sea line + m], or the first
            # non-sky thing from the top is an occluder (pole, mast) => skyline hidden here.
            # Occluders far above (sun flare, cloud clear of the land) do not matter.
            # land column: what matters is the silhouette (band around L1; the land must be seen
            # just below it). sea column: the band around the sea line.
            if np.isfinite(coarse_l1[i]) and first_cls[i] == LAND:
                y_a, y_b = coarse_l1[i] - 4 * cam_band, min(coarse_l1[i] + 3 * cam_band, Lrow[i] - 0.5 * cam_band)
            else:
                y_a, y_b = Lrow[i] - 4 * cam_band, Lrow[i] + cam_band
            a0 = int(max(0, y_a / sy)); a1 = int(min(SCENE_H, max(y_b, y_a + sy) / sy + 1))
            occ_here = (P[OCCL, a0:a1, c] > 0.5).any() or (first_cls[i] == OCCL and np.where(colc == OCCL)[0][0] * sy < Lrow[i] + cam_band)
            occl[i] = bool(occ_here)
            if occ_here:
                ids = A.obstacle_ids[a0:a1, c]
                ids = ids[ids > 0]
                occl_type[i] = id2type.get(int(np.bincount(ids).argmax()), "other") if len(ids) else "other"
                reasons[i] = "obstacle: " + occl_type[i]
                continue
            if top_land[i]:
                reasons[i] = "land reaches frame top"
                continue
            if open_sea[i]:
                st[i] = SEA_ST
                continue
            if first_cls[i] == LAND and np.isfinite(coarse_l1[i]):
                prof = heat_col(1, u[i])
                r, pk, spread = _topmost_peak(prof, coarse_l1[i] - y0, win=max(6, 2.5 * sy))
                if r is None:
                    reasons[i] = "no silhouette edge"
                    continue
                y = r + y0
                # land standing < 2 px above the sea line cannot be told from sea-horizon noise
                if y > Lrow[i] - 2.0 * W / 1600:
                    reasons[i] = "land/sea ambiguous at the sea line"
                    continue
                L1[i] = y
                L1raw[i] = spread / math.sqrt(max(pk, 0.05))
                contrast[i] = _contrast(imgf, u[i], y)
                if faint_gap[i] >= 3 * sy:
                    lower[i] = True
                # faint land at the threshold: prefer unknown to a guess (§5). Tuned on the
                # out-of-fold predictions (v0.1 log).
                if np.isfinite(contrast[i]) and (contrast[i] < 0.025 or (lower[i] and contrast[i] < 0.10)):
                    reasons[i] = "faint land (low contrast / haze)"
                    lower[i] = False
                    continue
                st[i] = LAND_ST
            elif first_cls[i] == SEA:
                reasons[i] = "sea horizon not visible (glare/spray/haze)"
            else:
                reasons[i] = "no land/sea found"

        # isolated short land/sea runs inside unknown -> unknown (they cannot fill a 0.5 deg bin)
        _drop_short_runs(st, reasons, minlen=3)
        for i in range(N_COLS):
            if st[i] != LAND_ST:
                L1[i] = np.nan if st[i] != LAND_ST else L1[i]
        A.state, A.reasons, A.L1, A.L1_raw_sigma = st, reasons, L1, L1raw
        A.contrast, A.lower_bound, A.occl, A.occl_type = contrast, lower, occl, occl_type

        # ---- L3 waterline: where land meets the sea (scene land->sea boundary, refined on
        # EdgeNet channel 3). Marked in every column where it is seen, open horizon or not.
        A.waterline, A.waterline_sigma_px = self._waterline(u, wl_edge, sc, occm, heat_col, y0, Hm, sy,
                                                            np.full(N_COLS, -np.inf) if no_line else Lrow)

        # ---- near-shore waterline: with no open sea and tall (near) land, the land/sea line sits a
        # distance-dependent amount below the true horizon and can be tilted (IMG_4976: 2.9 deg).
        if quality == "good" and src == "waterline":
            hgt = (Lrow - L1) / focal_px
            if np.isfinite(hgt).sum() >= 10 and math.degrees(np.nanpercentile(hgt, 90)) > 4.0:
                quality = "low"
                A.horizon_quality = "low"
                A.horizon_note = "near-shore waterline (land rises > 4 deg): may sit below / tilted from the horizon"
                okp = np.isfinite(L0obs)
                uc = float(u[okp].mean()) if okp.any() else W / 2
                A.line_sigma_px = np.hypot(max(sig, 0.03 * H), np.abs(u - uc) * math.tan(math.radians(3.0)))

        # ---- inner ridges (L2): EdgeNet ch2 peaks strictly between L1 and L0, chained
        A.inner1, A.inner2 = self._inner_ridges(u, L1, Lrow, st, heat_col, y0)

        # ---- sigma (px)
        self._src = src
        A.L1_sigma_px = self._sigma_px(L1raw, lower, faint_gap, contrast)
        A.sigma_includes_horizon = bool(self.calib)
        A.inner1_sigma_px = np.where(np.isfinite(A.inner1), 1.5, np.nan)
        A.inner2_sigma_px = np.where(np.isfinite(A.inner2), 1.5, np.nan)
        A.timings["total_s"] = time.time() - t0
        return A

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _fit_line(u, rows, thr):
        ok = np.isfinite(rows)
        if ok.sum() < 12:
            return None, None
        uu, rr = u[ok], rows[ok]
        best, bestin = None, 0
        rng = np.random.default_rng(0)
        for _ in range(200):
            i, j = rng.choice(len(uu), 2, replace=False)
            if abs(uu[j] - uu[i]) < 20:
                continue
            a = (rr[j] - rr[i]) / (uu[j] - uu[i]); b = rr[i] - a * uu[i]
            inl = np.abs(rr - (a * uu + b)) < thr
            if inl.sum() > bestin:
                bestin, best = inl.sum(), inl
        if best is None or bestin < 12:
            return None, None
        a, b = np.polyfit(uu[best], rr[best], 1)
        return (a, b), "sea"

    def _waterline(self, u, wl_edge, sc, occm, heat_col, y0, Hm, sy, Lrow):
        wl = np.full(N_COLS, np.nan)
        ws = np.full(N_COLS, np.nan)
        for i in np.where(np.isfinite(wl_edge))[0]:
            r = None
            if Hm.shape[0] >= 4:
                r, pk, spread = _peak_near(heat_col(3, u[i]), wl_edge[i] - y0, 3 * sy)
                if r is not None and pk > 0.25:
                    wl[i] = r + y0
                    ws[i] = max(0.7, spread / math.sqrt(max(pk, 0.05)))
                    continue
            wl[i] = wl_edge[i]          # scene boundary only: coarser
            ws[i] = 0.5 * sy
        # the sea surface is never above the sea horizon
        above = wl < Lrow - 2.0
        wl[above] = np.nan
        ws[above] = np.nan
        # drop isolated pieces (< 3 columns)
        ok = np.isfinite(wl)
        i = 0
        while i < N_COLS:
            if ok[i]:
                j = i
                while j < N_COLS and ok[j]:
                    j += 1
                if j - i < 3:
                    wl[i:j] = np.nan
                    ws[i:j] = np.nan
                i = j
            else:
                i += 1
        return wl, ws

    def _inner_ridges(self, u, L1, Lrow, st, heat_col, y0):
        cand = []
        for i in range(N_COLS):
            pk_list = []
            if st[i] == LAND_ST and np.isfinite(L1[i]) and Lrow[i] - L1[i] > 6:
                prof = heat_col(2, u[i])
                a0 = int(L1[i] - y0 + 3); a1 = int(Lrow[i] - y0 - 2)
                for r, p in _all_peaks(prof, a0, a1, thr=0.45):
                    pk_list.append(r + y0)
            cand.append(pk_list)
        # chain peaks across neighbouring columns; keep chains >= 6 columns
        chains = []  # list of dict col->row
        active = []
        for i, pks in enumerate(cand):
            new_active = []
            used = set()
            for ch in active:
                last = ch[max(ch)]
                if max(ch) != i - 1:
                    continue
                best = None
                for k, r in enumerate(pks):
                    if k in used:
                        continue
                    if abs(r - last) <= 3 and (best is None or abs(r - last) < abs(pks[best] - last)):
                        best = k
                if best is not None:
                    used.add(best); ch[i] = pks[best]; new_active.append(ch)
            for k, r in enumerate(pks):
                if k not in used:
                    ch = {i: r}; chains.append(ch); new_active.append(ch)
            active = new_active
        inner = [[] for _ in range(N_COLS)]
        for ch in chains:
            if len(ch) >= 10:
                for i, r in ch.items():
                    inner[i].append(r)
        i1 = np.full(N_COLS, np.nan); i2 = np.full(N_COLS, np.nan)
        for i, rs in enumerate(inner):
            rs = sorted(rs)
            if len(rs) >= 1:
                i1[i] = rs[0]
            if len(rs) >= 2:
                i2[i] = rs[1]
        return i1, i2

    def _sigma_px(self, raw, lower, gap, contrast):
        sig = np.full(N_COLS, np.nan)
        ok = np.isfinite(raw)
        if self.calib and "land" in self.calib:
            c = self.calib["land"].get(self._src, self.calib["land"].get("sea"))
            edges = np.array(c["edges"]); q = np.array(c["q68_px"])
            k = np.clip(np.searchsorted(edges, raw[ok]) - 1, 0, len(q) - 1)
            sig[ok] = q[k]
            lb = ok & lower
            sig[lb] = np.maximum(sig[lb] * c.get("lower_bound_factor", 2.0), gap[lb])
        else:
            sig[ok] = np.maximum(0.7, raw[ok])
            sig[ok & lower] = np.maximum(sig[ok & lower] * 2, gap[ok & lower])
        return sig


# ---------------------------------------------------------------------- helpers
def _subpix(prof, k):
    if 0 < k < len(prof) - 1:
        l, c, r = np.log(max(prof[k - 1], 1e-6)), np.log(max(prof[k], 1e-6)), np.log(max(prof[k + 1], 1e-6))
        den = l - 2 * c + r
        if den < 0:
            return k + float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))
    return float(k)


def _spread(prof, k, half=4):
    a, b = max(0, k - half), min(len(prof), k + half + 1)
    w = prof[a:b]
    ys = np.arange(a, b)
    if w.sum() <= 0:
        return 3.0
    m = (w * ys).sum() / w.sum()
    return float(math.sqrt(max((w * (ys - m) ** 2).sum() / w.sum(), 0.05)))


def _peak_near(prof, r0, win):
    a = int(max(0, math.floor(r0 - win))); b = int(min(len(prof), math.ceil(r0 + win) + 1))
    if b - a < 3:
        return None, 0.0, 0.0
    k = a + int(np.argmax(prof[a:b]))
    return _subpix(prof, k), float(prof[k]), _spread(prof, k)


def _all_peaks(prof, a, b, thr):
    a = max(1, a); b = min(len(prof) - 1, b)
    out = []
    for k in range(a, b):
        if prof[k] >= thr and prof[k] >= prof[k - 1] and prof[k] > prof[k + 1]:
            out.append((_subpix(prof, k), float(prof[k])))
    return out


def _topmost_peak(prof, r0, win, rel=0.45, absmin=0.2):
    """Topmost-wins: the highest (smallest row) local max within the window that is strong enough."""
    a = int(max(1, math.floor(r0 - win))); b = int(min(len(prof) - 1, math.ceil(r0 + win) + 1))
    if b - a < 3:
        return None, 0.0, 0.0
    seg = prof[a:b]
    m = float(seg.max())
    if m < absmin:
        return None, m, 0.0
    for k in range(a, b):
        if prof[k] >= max(absmin, rel * m) and prof[k] >= prof[k - 1] and prof[k] >= prof[k + 1]:
            return _subpix(prof, k), float(prof[k]), _spread(prof, k)
    k = a + int(np.argmax(seg))
    return _subpix(prof, k), m, _spread(prof, k)


def _contrast(imgf, u, y, d=4):
    x = int(round(u)); H = imgf.shape[0]
    a = imgf[max(0, int(y) - 2 - d): max(1, int(y) - 2), x]
    b = imgf[min(H - 1, int(y) + 2): min(H, int(y) + 2 + d), x]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(np.linalg.norm(a.mean(0) - b.mean(0)))


def _robust_line(u, rows, iters=6):
    ok = np.isfinite(rows)
    if ok.sum() < 6:
        return None
    uu, rr = u[ok], rows[ok]
    w = np.ones_like(uu)
    a, b = np.polyfit(uu, rr, 1)
    for _ in range(iters):
        res = rr - (a * uu + b)
        s = 1.4826 * np.median(np.abs(res)) + 1e-3
        w = 1.0 / np.maximum(1.0, np.abs(res) / (1.5 * s))
        w[np.abs(res) > 6 * s] = 0
        if w.sum() < 6:
            break
        a, b = np.polyfit(uu, rr, 1, w=np.sqrt(w))
    res = rr - (a * uu + b)
    keep = w > 0
    s = float(1.4826 * np.median(np.abs(res[keep]))) if keep.any() else 2.0
    return float(a), float(b), s, int(keep.sum())


def _drop_short_runs(st, reasons, minlen=3):
    n = len(st)
    i = 0
    while i < n:
        j = i
        while j < n and st[j] == st[i]:
            j += 1
        if st[i] != UNK_ST and j - i < minlen:
            left = st[i - 1] if i > 0 else UNK_ST
            right = st[j] if j < n else UNK_ST
            if left == UNK_ST and right == UNK_ST:
                for k in range(i, j):
                    st[k] = UNK_ST
                    reasons[k] = "isolated short run"
        i = j
