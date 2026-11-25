"""Train SceneNet / EdgeNet on hand-labelled photos.

  python -m skyliner.train --data example_dataset --net scene --fold 0 --folds 4 --steps 2500
  python -m skyliner.train --data example_dataset --net edge  --fold all --steps 4000

Folds are contiguous blocks of photo ids (photos taken seconds apart are near-duplicates,
so a random split would leak). fold=all trains on every photo (the shipped model).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .labels import IGNORE, OCCL, build_targets, heatmaps, load_dataset
from .models import SCENE_H, SCENE_W, EdgeNet, SceneNet

CACHE_SCALE = 0.5  # scene cache resolution relative to the photo


def fold_split(samples, fold, folds):
    if fold == "all":
        return samples, []
    k = int(fold)
    n = len(samples)
    a, b = round(k * n / folds), round((k + 1) * n / folds)
    return samples[:a] + samples[b:], samples[a:b]


# ------------------------------------------------------------------ augmentation
def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    x = img.astype(np.float32)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        m = x.mean((0, 1), keepdims=True)
        x = (x - m) * rng.uniform(0.6, 1.3) + m
    if rng.random() < 0.4:  # saturation
        g = x.mean(2, keepdims=True)
        x = (x - g) * rng.uniform(0.5, 1.4) + g
    if rng.random() < 0.3:  # haze veil toward a sky-ish colour (mild, keeps edges)
        veil = np.array([rng.uniform(150, 230), rng.uniform(170, 235), rng.uniform(200, 255)], np.float32)
        a = rng.uniform(0.05, 0.3)
        x = x * (1 - a) + veil * a
    x = np.clip(x, 0, 255)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.3:
        x = x + np.random.normal(0, rng.uniform(1, 6), x.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def synthetic_bar(img, cls, rng):
    """Paint a pole/railing-like occluder (straight thick bar, shaded)."""
    H, W = cls.shape
    w = rng.uniform(0.01, 0.12) * W
    x0 = rng.uniform(-0.1, 1.1) * W
    slant = rng.uniform(-0.25, 0.25) * H
    vertical = rng.random() < 0.7
    if vertical:
        pts = np.array([[x0, 0], [x0 + w, 0], [x0 + w + slant, H], [x0 + slant, H]])
    else:  # railing across the lower/mid frame
        y0 = rng.uniform(0.3, 1.0) * H
        pts = np.array([[0, y0], [W, y0 + slant * 0.3], [W, y0 + slant * 0.3 + w * 0.4], [0, y0 + w * 0.4]])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [pts.astype(np.int32)], 1)
    base = np.array(rng.choice([[235, 235, 235], [60, 65, 80], [130, 135, 140], [30, 50, 110], [200, 200, 205]]), np.float32)
    ramp = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.8, 1.2), W)[None, :, None]
    paint = np.clip(base[None, None, :] * ramp, 0, 255)
    sel = m > 0
    img[sel] = paint.repeat(H, 0)[sel].astype(np.uint8)
    cls[sel] = OCCL


def _smooth_noise(h, w, rng, octaves=4):
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for o in range(octaves):
        gh, gw = max(2, h >> (4 - o)), max(2, w >> (4 - o))
        g = np.random.default_rng(rng.randrange(1 << 30)).random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= 0.5
    out -= out.min()
    return out / max(out.max(), 1e-6)


def synthetic_cloud(img, cls, rng):
    """Cumulus/fog-like blob sitting on the land silhouette (or near the sea line), labelled
    occluder: the model must learn that cloud touching land hides the true skyline."""
    from .labels import LAND
    H, W = cls.shape
    land_cols = np.where((cls == LAND).any(0))[0]
    if len(land_cols) == 0:
        return
    cx = int(land_cols[rng.randrange(len(land_cols))])
    col = np.where(cls[:, cx] == LAND)[0]
    top = int(col[0])
    w = int(rng.uniform(0.06, 0.35) * W)
    h = int(w * (rng.uniform(0.12, 0.3) if rng.random() < 0.5 else rng.uniform(0.3, 0.7)))  # cap cloud | cumulus
    cy = top + int(rng.uniform(-0.6, 0.3) * h)          # straddles the summit
    x0, y0 = cx - w // 2, cy - h // 2
    n = _smooth_noise(h, w, rng, octaves=3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ell = 1 - (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    # real clouds over the Aegean: bright, smooth, mostly opaque with a soft rim (grey translucent
    # blobs look like hazy land or pale rock and teach the wrong thing)
    a = np.clip((n * 0.6 + ell * 1.0 - 0.35) * rng.uniform(3.0, 6.0), 0, 1) * rng.uniform(0.88, 1.0)
    shade = np.clip(0.93 + 0.06 * n - 0.05 * (yy / h), 0, 1)[..., None]
    g = rng.uniform(238, 252)
    base = np.array([g, g + rng.uniform(0, 2), g + rng.uniform(0, 4)], np.float32)  # neutral white
    patch = base * shade
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(H, y0 + h), min(W, x0 + w)
    if ys1 <= ys0 or xs1 <= xs0:
        return
    from .labels import SEA
    a = a.copy()
    # clouds never sit below the sea line: clip at the first sea row of each column
    for j in range(xs0, xs1):
        sea = np.where(cls[:, j] == SEA)[0]
        if len(sea):
            cut = sea[0] - y0
            if cut < h:
                a[max(cut, 0):, j - x0] = 0
    sa = a[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0, None]
    sp = patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    reg = img[ys0:ys1, xs0:xs1].astype(np.float32)
    img[ys0:ys1, xs0:xs1] = np.clip(reg * (1 - sa) + sp * sa, 0, 255).astype(np.uint8)
    m = sa[..., 0] > 0.5
    sub = cls[ys0:ys1, xs0:xs1]
    sub[m & (sub != 255)] = OCCL


def paste_occluder(img, cls, src_img, src_occ, rng):
    H, W = cls.shape
    s_img, s_m = src_img, src_occ
    if rng.random() < 0.5:
        s_img, s_m = s_img[:, ::-1], s_m[:, ::-1]
    dx = int(rng.uniform(-0.6, 0.6) * W)
    dy = int(rng.uniform(-0.25, 0.25) * H)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    wi = cv2.warpAffine(np.ascontiguousarray(s_img), M, (W, H))
    wm = cv2.warpAffine(np.ascontiguousarray(s_m), M, (W, H), flags=cv2.INTER_NEAREST)
    sel = wm > 0
    img[sel] = wi[sel]
    cls[sel] = OCCL


# ------------------------------------------------------------------ scene data
class SceneData:
    crop = (384, 256)  # train on crops of the 512x384 scene frame (fully convolutional)

    def __init__(self, samples, seed=0):
        self.rng = random.Random(seed)
        self.items = []
        self.occ_src = []
        for s in samples:
            T = build_targets(s)
            img = s.image()
            W, H = int(s.W * CACHE_SCALE), int(s.H * CACHE_SCALE)
            im = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            cl = cv2.resize(T.cls, (W, H), interpolation=cv2.INTER_NEAREST)
            self.items.append((im, cl))
            if (cl == OCCL).any():
                self.occ_src.append((im, (cl == OCCL).astype(np.uint8)))

    def sample(self):
        rng = self.rng
        im, cl = self.items[rng.randrange(len(self.items))]
        im, cl = im.copy(), cl.copy()
        if self.occ_src and rng.random() < 0.35:
            si, sm = self.occ_src[rng.randrange(len(self.occ_src))]
            if si.shape == im.shape:
                paste_occluder(im, cl, si, sm, rng)
        if rng.random() < 0.25:
            synthetic_bar(im, cl, rng)
        if rng.random() < 0.15:
            synthetic_cloud(im, cl, rng)
        H, W = cl.shape
        ang = rng.uniform(-4, 4)
        zoom = rng.uniform(1.0, 1.35)
        sx = SCENE_W / W * zoom
        cx, cy = W / 2 + rng.uniform(-0.15, 0.15) * W, H / 2 + rng.uniform(-0.2, 0.2) * H
        M = cv2.getRotationMatrix2D((cx, cy), ang, sx)
        TW, TH = self.crop
        M[0, 2] += TW / 2 - cx
        M[1, 2] += TH / 2 - cy
        if rng.random() < 0.5:
            M = np.float32([[-1, 0, TW - 1], [0, 1, 0]]) @ np.vstack([M, [0, 0, 1]])
        x = cv2.warpAffine(im, M[:2], (TW, TH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        y = cv2.warpAffine(cl, M[:2], (TW, TH), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
        x = photometric(x, rng)
        return x, y

    def batch(self, n):
        xs, ys = zip(*[self.sample() for _ in range(n)])
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float()
        y = torch.from_numpy(np.stack(ys)).long()
        return x, y


# ------------------------------------------------------------------ edge data
EDGE_CROP = 128


class EdgeData:
    def __init__(self, samples, seed=0):
        self.rng = random.Random(seed)
        self.items = []
        for s in samples:
            T = build_targets(s)
            occ = s.occluder_mask()
            self.items.append((s.image(), T, occ))

    def sample(self):
        rng = self.rng
        img, T, occ = self.items[rng.randrange(len(self.items))]
        H, W = img.shape[:2]
        r = rng.random()
        lvl = "L1" if r < 0.45 else ("L0vis" if r < 0.65 else ("L0land" if r < 0.82 else rng.choice(["L2a", "L2b"])))
        rows = T.rows[lvl]
        cand = np.where(np.isfinite(rows) & T.edge_valid)[0]
        if len(cand) == 0 or rng.random() < 0.12:
            band_top = np.nanmin(T.rows["L1"]) if np.isfinite(T.rows["L1"]).any() else np.nanmin(T.rows["L0"])
            cx = rng.uniform(0, W)
            cy = rng.uniform(band_top - 120, np.nanmax(T.rows["L0"]) + 120)
        else:
            cx = cand[rng.randrange(len(cand))]
            cy = rows[int(cx)] + rng.uniform(-50, 50)
        big = int(EDGE_CROP * 1.5)
        x0 = int(np.clip(cx - big / 2, 0, W - big))
        y0 = int(np.clip(cy - big / 2, 0, H - big))
        crop = img[y0:y0 + big, x0:x0 + big]
        hm = heatmaps(T.rows, y0, y0 + big, x0, x0 + big, sigma=1.0)
        colv = T.edge_valid[x0:x0 + big][None, :].repeat(big, 0)
        oc = occ[y0:y0 + big, x0:x0 + big] > 0
        valid = (colv | oc).astype(np.float32)
        hm[:, oc] = 0.0
        ang = rng.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D((big / 2, big / 2), ang, 1.0)
        off = (big - EDGE_CROP) / 2
        M[0, 2] -= off
        M[1, 2] -= off
        if rng.random() < 0.5:
            M = np.float32([[-1, 0, EDGE_CROP - 1], [0, 1, 0]]) @ np.vstack([M, [0, 0, 1]])
            M = M[:2]
        w = lambda a, fl=cv2.INTER_LINEAR, bv=0: cv2.warpAffine(a, M, (EDGE_CROP, EDGE_CROP), flags=fl, borderValue=bv)
        x = photometric(w(crop), rng)
        hm = np.stack([w(h) for h in hm])
        valid = w(valid, cv2.INTER_NEAREST, 0)
        return x, hm, valid

    def batch(self, n):
        xs, hs, vs = zip(*[self.sample() for _ in range(n)])
        x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float()
        return x, torch.from_numpy(np.stack(hs)), torch.from_numpy(np.stack(vs))


def edge_loss(logits, hm, valid):
    # weighted BCE: strong weight on the ridge of the gaussian, light on far negatives;
    # L2 (inner ridges) is incompletely labelled -> its negatives count less.
    w = 1.0 + 25.0 * hm
    w[:, 2] = torch.where(hm[:, 2] > 0.05, w[:, 2], 0.3 * w[:, 2])
    l = F.binary_cross_entropy_with_logits(logits, hm, reduction="none") * w * valid[:, None]
    return l.sum() / (valid.sum() * logits.shape[1] + 1)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--net", choices=["scene", "edge"], required=True)
    ap.add_argument("--fold", default="all")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    samples = load_dataset(a.data)
    tr, va = fold_split(samples, a.fold, a.folds)
    print(f"{a.net} fold={a.fold}: train {len(tr)} val {len(va)}", flush=True)
    t0 = time.time()
    data = SceneData(tr) if a.net == "scene" else EdgeData(tr)
    print(f"data ready {time.time() - t0:.0f}s", flush=True)
    net = SceneNet() if a.net == "scene" else EdgeNet()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    cw = torch.tensor([1.0, 1.5, 1.0, 2.0])
    net.train()
    t0 = time.time()
    run = 0.0
    for step in range(1, a.steps + 1):
        if a.net == "scene":
            x, y = data.batch(a.batch)
            loss = F.cross_entropy(net(x), y, weight=cw, ignore_index=IGNORE)
        else:
            x, hm, v = data.batch(int(a.batch * 1.5))
            loss = edge_loss(net(x), hm, v)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 50 == 0 or step == a.steps:
            el = time.time() - t0
            print(f"step {step}/{a.steps} loss {run:.4f}  {el:.0f}s  eta {el / step * (a.steps - step):.0f}s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.net}_{a.fold}.pt")
    torch.save(net.state_dict(), path)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
