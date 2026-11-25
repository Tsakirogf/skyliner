"""Obstacles: everything that hides the scene and is not in the DEM (ship structure, other vessels,
clouds, glare, wake/spray). SceneNet segments them as one class; here they are split into
instances and given a type from geometry and colour (heuristic: the network has one class).

Types:
  ship_structure  touches a frame edge and reaches below the sea line, or is large (own ship: masts,
                  railings, machinery, pillars)
  vessel_object   small, near the sea line, clear of the frame edges (other ships, buoys, rocks)
  cloud           above the sea line, whitish, clear of the frame bottom (cloud or fog on land)
  glare           very bright, low saturation (sun flare in the sky, sun glitter on the sea)
  wake_spray      below the sea line, whitish, on the sea
  other           none of the above
"""
from __future__ import annotations

import cv2
import numpy as np

TYPES = ["ship_structure", "vessel_object", "cloud", "glare", "wake_spray", "other"]
TYPE_ID = {t: i + 1 for i, t in enumerate(TYPES)}  # 0 = no obstacle


def full_res_mask(p_occ: np.ndarray, W: int, H: int, thr: float = 0.5) -> np.ndarray:
    """Upsample the SceneNet obstacle probability to the photo and threshold (uint8 0/1)."""
    p = cv2.resize(p_occ.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return (p > thr).astype(np.uint8)


def instances(p_occ: np.ndarray, img: np.ndarray, line_row_at, min_area_scene: int = 20,
              group_px: int = 5) -> tuple[list, np.ndarray]:
    """Connected components of the scene-resolution obstacle mask.

    p_occ: SceneNet obstacle probability (h x w). img: photo RGB (H x W x 3).
    line_row_at(x_photo) -> sea-line row in photo pixels (or None when unknown).
    Returns (list of dicts, label image at scene resolution with instance ids).
    """
    h, w = p_occ.shape
    H, W = img.shape[:2]
    sx, sy = W / w, H / h
    m = (p_occ > 0.5).astype(np.uint8)
    # group pieces of one object (machinery seen through gaps, a dish on its mount): label the
    # dilated mask, keep only the original obstacle pixels
    grown = cv2.dilate(m, np.ones((2 * group_px + 1, 2 * group_px + 1), np.uint8))
    n, glab = cv2.connectedComponents(grown, connectivity=8)
    lab = np.where(m > 0, glab, 0).astype(np.int32)
    stats = np.zeros((n, 5), np.int64)
    cent = np.zeros((n, 2))
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if len(xs):
            stats[i] = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, len(xs)]
            cent[i] = [xs.mean(), ys.mean()]
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
    out = []
    keep = np.zeros_like(lab)
    k = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_scene:
            continue
        k += 1
        sel = lab == i
        keep[sel] = k
        V = hsv[..., 2][sel].mean() / 255.0
        S = hsv[..., 1][sel].mean() / 255.0
        cx, cy = cent[i]
        X0, Y0, X1, Y1 = x * sx, y * sy, (x + bw) * sx, (y + bh) * sy
        lr = line_row_at((cx + 0.5) * sx) if line_row_at else None
        touches = {"bottom": y + bh >= h, "top": y <= 0, "left": x <= 0, "right": x + bw >= w}
        frac = area / float(h * w)
        below_line = lr is not None and (cy + 0.5) * sy > lr
        reaches_below = lr is not None and Y1 > lr + 2 * sy
        if V > 0.88 and S < 0.12 and frac < 0.15:
            typ = "glare"
        elif (touches["bottom"] or touches["left"] or touches["right"]) and (reaches_below or lr is None) or frac > 0.08:
            typ = "ship_structure"
        elif below_line and S < 0.35 and V > 0.45:
            typ = "wake_spray"
        elif lr is not None and not below_line and S < 0.25 and V > 0.6 and not touches["bottom"]:
            typ = "cloud"
        elif frac < 0.01 and lr is not None and abs((cy + 0.5) * sy - lr) < 0.08 * H:
            typ = "vessel_object"
        else:
            typ = "other"
        out.append({
            "id": k, "type": typ,
            "bbox_px": [round(X0, 1), round(Y0, 1), round(X1, 1), round(Y1, 1)],
            "area_frac": round(frac, 5),
            "mean_value": round(float(V), 3), "mean_saturation": round(float(S), 3),
            "touches_edges": [e for e, t in touches.items() if t],
        })
    # parts of the own ship seen through gaps (grating, rails) are disconnected in the mask:
    # anything whose box touches a ship_structure box (expanded by 4 % of the frame) and does not
    # sit above it in the sky belongs to the ship too
    for _ in range(3):
        ships = [o["bbox_px"] for o in out if o["type"] == "ship_structure"]
        changed = False
        for o in out:
            if o["type"] in ("ship_structure", "glare", "cloud"):
                continue
            x0, y0, x1, y1 = o["bbox_px"]
            for sx0, sy0, sx1, sy1 in ships:
                mx, my = 0.04 * W, 0.04 * H
                if x0 < sx1 + mx and x1 > sx0 - mx and y0 < sy1 + my and y1 > sy0 - my and y1 > sy0 - my:
                    o["type"] = "ship_structure"
                    changed = True
                    break
        if not changed:
            break
    return out, keep
