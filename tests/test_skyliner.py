"""Run: python -m pytest tests  (or: python tests/test_skyliner.py)

Needs example_dataset/ next to the package (set SKYLINER_DATA to override) and, for the
annotation test, the ONNX models in models/.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.environ.get("SKYLINER_DATA", os.path.join(ROOT, "example_dataset"))

from skyliner.camera import Camera, dip_deg  # noqa: E402
from skyliner.labels import load_dataset  # noqa: E402


def test_dip():
    assert abs(dip_deg(2.0) - (-0.042)) < 0.001
    assert abs(dip_deg(20.0) - (-0.1335)) < 0.002


def test_camera_matches_hand_observations():
    """Our §4.2 conversion reproduces the hand-label observation.json land elevations."""
    diffs = []
    for s in load_dataset(DATA):
        if not s.obs:
            continue
        m = s.meta
        cam = Camera(m["width"], m["height"], m["focal_px"], m["horizon_a"], m["horizon_b"])
        out = s.level("outer")
        land = np.array(s.obs["state"]) == "land"
        _, el = cam.pixel_to_angles(s.col_x(), np.where(np.isfinite(out), out, cam.horizon_row(s.col_x())))
        if land.any():
            diffs.append(np.median(np.abs(el[land] - np.array(s.obs["elev_deg"])[land])))
        assert abs(cam.roll_deg - s.obs["roll_deg"]) < 1e-6
        assert abs(cam.pitch_deg - s.obs["pitch_deg"]) < 0.01
    assert np.median(diffs) < 0.002


def test_annotate_contract_rules():
    models = os.path.join(ROOT, "models")
    if not os.path.exists(os.path.join(models, "scene.onnx")):
        return
    from skyliner.__main__ import main
    ids = sorted(os.listdir(DATA))[:2]
    with tempfile.TemporaryDirectory() as inp, tempfile.TemporaryDirectory() as out:
        for i in ids:
            os.symlink(os.path.join(DATA, i), os.path.join(inp, i))
        main(["annotate", inp, out, "--models", models, "--no-overlay"])
        for i in ids:
            o = json.load(open(os.path.join(out, i, "observation.json")))
            assert o["schema"] == "horizon.observation/2+json"
            for k in ("id", "platform_id", "t_start_ns", "t_end_ns", "frame", "az_deg", "elev_deg", "state", "source"):
                assert k in o, k
            az = np.array(o["az_deg"])
            assert np.all(np.diff(az) > 0) and az[-1] - az[0] < 360
            n = len(az)
            assert len(o["elev_deg"]) == n and len(o["state"]) == n and len(o["sigma_deg"]) == n
            assert set(o["state"]) <= {"sea", "land", "unknown"}
            assert all(math.isfinite(e) for e in o["elev_deg"])           # unknown: anything finite
            for st, sg in zip(o["state"], o["sigma_deg"]):
                assert (sg == float("inf")) == (st == "unknown")
            assert 1 <= len(o["id"]) <= 64
            for L in o["layers"]:
                assert len(L["elev_deg"]) == n
            assert {L["kind"] for L in o["layers"]} == {"inner_ridge", "waterline"}
            ob = json.load(open(os.path.join(out, i, "obstacles.json")))
            assert all(x["type"] in ob["types"] for x in ob["obstacles"])
            assert os.path.exists(os.path.join(out, i, "obstacle_mask.png"))
            lab = json.load(open(os.path.join(out, i, "label.json")))
            assert len(lab["waterline"]) == 512 and len(lab["occl_type"]) == 512
            assert not os.path.exists(os.path.join(out, i, "rejected.json"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
