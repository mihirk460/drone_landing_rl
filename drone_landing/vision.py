"""Find the landing pad's "H" in a downward camera image and turn it into a world-frame estimate."""
import math

import cv2
import numpy as np


def find_h_centroid(rgb: np.ndarray, dark_threshold: int = 60, min_pixels: int = 3):
    """Locate the black H on the white pad.

    The H is the only near-black thing in the scene (ground is green, pad is white,
    the drone itself is out of the camera's view), so a threshold + largest blob is enough.

    Returns (u, v, pixel_count) with u, v in [-1, 1] normalised image coordinates
    (u right, v up, (0,0) = image centre), or None if nothing is found.
    """
    h, w = rgb.shape[:2]
    mask = (rgb.max(axis=2) < dark_threshold).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    # label 0 is background; pick the largest remaining blob
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < min_pixels:
        return None
    cx, cy = centroids[idx]
    u = (cx - (w - 1) / 2.0) / ((w - 1) / 2.0)
    v = -(cy - (h - 1) / 2.0) / ((h - 1) / 2.0)
    return float(u), float(v), area


def pixel_to_ground(u: float, v: float, cam_pos: np.ndarray, cam_mat: np.ndarray,
                    fovy_deg: float, aspect: float, ground_z: float):
    """Cast a ray through normalised pixel (u, v) and intersect it with the plane z = ground_z.

    cam_mat is the 3x3 camera rotation (world <- camera). MuJoCo cameras look along -Z_cam,
    with +X_cam right and +Y_cam up in the image.
    Returns world xyz of the hit, or None if the ray does not hit the plane below.
    """
    t = math.tan(math.radians(fovy_deg) / 2.0)
    d_cam = np.array([u * t * aspect, v * t, -1.0])
    d_world = cam_mat @ d_cam
    if d_world[2] >= -1e-6:
        return None
    s = (ground_z - cam_pos[2]) / d_world[2]
    return cam_pos + s * d_world


def ground_to_pixel(p: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
                    fovy_deg: float, aspect: float):
    """Inverse of pixel_to_ground: project world point p to normalised (u, v).

    Returns None if the point is behind the camera or outside the image.
    """
    t = math.tan(math.radians(fovy_deg) / 2.0)
    d_cam = cam_mat.T @ (p - cam_pos)
    if d_cam[2] >= -1e-6:
        return None
    u = d_cam[0] / (-d_cam[2]) / (t * aspect)
    v = d_cam[1] / (-d_cam[2]) / t
    if abs(u) > 1.0 or abs(v) > 1.0:
        return None
    return float(u), float(v)
