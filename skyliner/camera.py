"""Pinhole camera + sea-horizon attitude, exactly as SKYLINER_REQUIREMENTS.md §4.2.

All pixel coordinates are in the photo's own pixel grid (width x height of the JPEG).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_R_M = 6_371_000.0
REFRACTION_K = 0.13
DEFAULT_EYE_M = 2.0


def dip_rad(eye_height_m: float = DEFAULT_EYE_M) -> float:
    """Sea-horizon dip below the true horizontal (negative), radians."""
    return -math.sqrt(2.0 * eye_height_m * (1.0 - REFRACTION_K) / EARTH_R_M)


def dip_deg(eye_height_m: float = DEFAULT_EYE_M) -> float:
    return math.degrees(dip_rad(eye_height_m))


@dataclass
class Camera:
    width: int
    height: int
    focal_px: float
    horizon_a: float  # sea horizon v = a*u + b (pixels)
    horizon_b: float
    eye_height_m: float = DEFAULT_EYE_M

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    @property
    def roll_rad(self) -> float:
        return math.atan(self.horizon_a)

    @property
    def pitch_rad(self) -> float:
        a, b, f = self.horizon_a, self.horizon_b, self.focal_px
        return math.atan2((a * self.cx + b - self.cy) * math.cos(self.roll_rad), f) + dip_rad(self.eye_height_m)

    @property
    def roll_deg(self) -> float:
        return math.degrees(self.roll_rad)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self.pitch_rad)

    def pixel_to_angles(self, u, v):
        """(u, v) pixels -> (azimuth offset, elevation) in degrees, camera frame."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        r, p, f = self.roll_rad, self.pitch_rad, self.focal_px
        x = u - self.cx
        y = v - self.cy
        xr = x * math.cos(r) + y * math.sin(r)       # de-roll
        yr = -x * math.sin(r) + y * math.cos(r)
        fwd = f * math.cos(p) + yr * math.sin(p)      # rotate by pitch
        up = -yr * math.cos(p) + f * math.sin(p)
        az = np.degrees(np.arctan2(xr, fwd))
        el = np.degrees(np.arctan2(up, np.hypot(xr, fwd)))
        return az, el

    def deg_per_px(self) -> float:
        return math.degrees(1.0 / self.focal_px)

    def horizon_row(self, u):
        return self.horizon_a * np.asarray(u, float) + self.horizon_b
