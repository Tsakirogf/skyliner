"""Camera profiles: focal length in pixels from EXIF, with a per-model calibration factor.

Requirement §4.2: residual focal scale error <= 0.3 %. The iPhone 13 main camera needed x1.031
over the EXIF value. Add other cameras here after calibrating them.
"""
from __future__ import annotations

import math

PROFILES = {
    # model substring : focal scale over EXIF 35 mm-equivalent focal length
    "iPhone 13": {"focal_scale": 1.031},
}


def focal_from_exif(path: str, width: int, height: int) -> dict | None:
    try:
        from PIL import ExifTags, Image
        im = Image.open(path)
        ex = im.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in {**dict(ex), **dict(ex.get_ifd(0x8769))}.items()}
    except Exception:
        return None
    f35 = tags.get("FocalLengthIn35mmFilm")
    model = str(tags.get("Model") or "")
    if not f35:
        return None
    scale = 1.0
    for k, p in PROFILES.items():
        if k in model:
            scale = p["focal_scale"]
    # 35 mm equivalent is defined on the 43.27 mm diagonal
    f_px = float(f35) * math.hypot(width, height) / 43.267 * scale
    return {"focal_px": f_px, "time": tags.get("DateTimeOriginal"), "sensor_id": f"visible:{model}" if model else None,
            "calibrated": scale != 1.0}
