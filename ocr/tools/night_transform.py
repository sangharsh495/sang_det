"""Turn daylight plate crops into ambient-night and 850 nm IR versions.

Matches the look of the indian-plate-nv-3cond set (IMX462 sensor):

ambient  street lighting, high analogue gain: monochrome, few photons, so
         Poisson shot noise + read noise, auto-exposure brings it back to mid
         grey, slight softening, JPEG.
ir       IR-cut filter out, 850 nm illuminator: monochrome, retroreflective
         plate face goes bright (often clipping and thinning the characters),
         surroundings fall dark, illuminator hotspot, light noise.

Usage:
    python night_transform.py SRC_DIR OUT_ROOT [--seed 0]
writes OUT_ROOT/ambient/<name> and OUT_ROOT/ir/<name> for every jpg in SRC_DIR.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _lin(img):
    x = img.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    y = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    return (y * 255).astype(np.uint8)


def _mono(lin_bgr, weights):
    return lin_bgr[..., 0] * weights[0] + lin_bgr[..., 1] * weights[1] + lin_bgr[..., 2] * weights[2]


def _jpeg(img, rng, lo=45, hi=85):
    q = int(rng.integers(lo, hi))
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def ambient(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lin = _lin(img)
    # Sodium (orange) or white LED street light, then a mono sensor.
    w = (0.10, 0.45, 0.45) if rng.random() < 0.5 else (0.25, 0.45, 0.30)
    y = _mono(lin, w)
    full_well = rng.uniform(20, 150)          # electrons at white: few photons
    e = rng.poisson(np.clip(y, 0, None) * full_well).astype(np.float32)
    e += rng.normal(0, rng.uniform(1.5, 4.0), e.shape)   # read noise
    y = e / full_well
    y = y / max(np.percentile(y, 99), 1e-3) * rng.uniform(0.75, 0.95)  # auto-exposure
    y = cv2.GaussianBlur(y, (0, 0), rng.uniform(0.4, 1.1))
    out = cv2.cvtColor(_srgb(y), cv2.COLOR_GRAY2BGR)
    return _jpeg(out, rng)


def ir(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lin = _lin(img)
    y = _mono(lin, (0.33, 0.33, 0.34))
    h, w = y.shape
    # Retroreflective sheeting: bright areas return far more light.
    bright = np.clip((y - np.percentile(y, 40)) / 0.4, 0, 1)
    boost = rng.uniform(2.5, 5.0)
    y = y * (0.35 + 0.65 * (1 - bright)) + bright * y * boost
    # Illuminator hotspot.
    cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2
    y = y * (0.55 + 0.45 * np.exp(-r2 / rng.uniform(0.08, 0.4)))
    # Bloom from clipped areas thins the characters.
    clip = np.clip(y - 0.9, 0, None)
    y = y + cv2.GaussianBlur(clip, (0, 0), rng.uniform(0.8, 2.0)) * rng.uniform(0.5, 1.5)
    y = y + rng.normal(0, rng.uniform(0.005, 0.03), y.shape)
    y = cv2.GaussianBlur(y, (0, 0), rng.uniform(0.3, 0.9))
    out = cv2.cvtColor(_srgb(y), cv2.COLOR_GRAY2BGR)
    return _jpeg(out, rng, 55, 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--list', help='file of image paths instead of SRC_DIR/*.jpg')
    a = ap.parse_args()
    paths = ([Path(l.strip()) for l in open(a.list) if l.strip()] if a.list
             else sorted(Path(a.src).glob('*.jpg')))
    for cond in ('ambient', 'ir'):
        (Path(a.out) / cond).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        rng = np.random.default_rng(a.seed * 1_000_003 + i)
        cv2.imwrite(str(Path(a.out) / 'ambient' / p.name), ambient(img, rng))
        cv2.imwrite(str(Path(a.out) / 'ir' / p.name), ir(img, rng))
    print(f'{len(paths)} crops -> {a.out}/ambient, {a.out}/ir')


if __name__ == '__main__':
    main()
