"""Generate the landing-pad texture: a white square with a black letter H.

Run once (the PNG is committed, so you normally don't need to):
    python scripts/make_pad_texture.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "assets" / "h_pad.png"


def make_h_pad(size: int = 256, margin: float = 0.14, bar: float = 0.20) -> np.ndarray:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    m = int(size * margin)
    b = int(size * bar)
    # two vertical bars
    img[m : size - m, m : m + b] = 0
    img[m : size - m, size - m - b : size - m] = 0
    # horizontal cross bar
    c = size // 2
    img[c - b // 2 : c + b // 2, m : size - m] = 0
    # thin black border so the pad edge is visible against the ground
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (0, 0, 0), 2)
    return img


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), make_h_pad())
    print(f"wrote {OUT}")
