"""Surface sampling and fixed screen-space marker style for real-first-frame rigid conditions."""

import cv2
import numpy as np
from scipy.spatial import cKDTree
import trimesh

SCENE_POINT_BUDGET = 640
MIN_POINTS_PER_ACTOR = 16
REFERENCE_WIDTH = 640
MARKER_RADIUS_PX = 1.10
REFERENCE_AREA_PER_POINT = 12
PROFILE = "common_surface_density_fixed_marker_v1"


def farthest_surface_points(mesh, count, seed):
    """Cover the mesh from area-weighted candidates, without copying surface vertices."""
    candidates, _ = trimesh.sample.sample_surface(mesh, max(256, 32 * count), seed=seed)
    candidates = np.unique(candidates, axis=0)
    if len(candidates) < count:
        raise ValueError("Too few distinct mesh samples")
    center = np.average(mesh.triangles_center, weights=mesh.area_faces, axis=0)
    index = int(np.linalg.norm(candidates - center, axis=1).argmax())
    distances = np.full(len(candidates), np.inf)
    samples = np.empty((count, 3))
    for i in range(count):
        samples[i] = candidates[index]
        delta = candidates - candidates[index]
        np.minimum(distances, np.einsum("ij,ij->i", delta, delta), out=distances)
        index = int(distances.argmax())
    return samples


def sample_rigid_surfaces(meshes, visible_pixels, image_width):
    areas = np.asarray([mesh.area for mesh in meshes])
    if len(areas) == 0 or not np.all(np.isfinite(areas) & (areas > 0)):
        raise ValueError("Invalid rigid surface areas")
    normalized_pixels = visible_pixels * (REFERENCE_WIDTH / image_width) ** 2
    budget = min(SCENE_POINT_BUDGET, max(1.0, normalized_pixels / REFERENCE_AREA_PER_POINT))
    # Resolve small participating bodies by increasing the shared density.
    # Giving only the small actor extra points would make its density inconsistent.
    density = max(budget / areas.sum(), MIN_POINTS_PER_ACTOR / areas.min())
    spacing = float(density ** -0.5)
    counts = np.maximum(1, np.rint(areas * density).astype(int))
    points, stats = [], []
    for i, (mesh, count) in enumerate(zip(meshes, counts, strict=True)):
        samples = farthest_surface_points(mesh, count, seed=314159 + i)
        neighbors = cKDTree(samples).query(samples, k=2)[0][:, 1] if count > 1 else None
        points.append(samples)
        stats.append(dict(
            actor=i, area=float(areas[i]), count=int(count), requested=int(count),
            common_spacing=spacing, density=float(count / areas[i]),
            nn_median=None if neighbors is None else float(np.median(neighbors)),
            nn_min=None if neighbors is None else float(neighbors.min()),
            nn_cv=None if neighbors is None else float(neighbors.std() / neighbors.mean()),
        ))
    return points, stats


def fixed_marker_image(points, camera_k, shape, near, far):
    """Project all material points onto one viewport, with a common marker radius."""
    height, width = shape
    supersample = 4
    canvas = np.zeros((height * supersample, width * supersample), np.uint8)
    z = points[:, 2]
    uv = np.full((len(points), 2), np.nan)
    valid = z > .01
    uv[valid] = (points[valid] @ camera_k.T)[:, :2] / z[valid, None]
    valid &= np.isfinite(uv).all(1)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    gray = np.rint(np.clip(90 + 165 * (far - z) / max(far - near, 1e-6), 90, 255)).astype(np.uint8)
    shift = 4
    precision = 1 << shift
    radius = max(1, round(MARKER_RADIUS_PX * width / REFERENCE_WIDTH * supersample * precision))
    for i in np.argsort(z)[::-1]:
        if valid[i]:
            center = tuple(np.rint((uv[i] * supersample - .5) * precision).astype(int))
            cv2.circle(canvas, center, radius, int(gray[i]), -1, cv2.LINE_8, shift)
    image = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
