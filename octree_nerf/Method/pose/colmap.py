import numpy as np
import trimesh

from octree_nerf.Method.colmap_io import *


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def center_poses(poses, pts3d=None, enable_cam_center=False):
    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-10)

    if pts3d is None or enable_cam_center:
        center = poses[:, :3, 3].mean(0)
    else:
        center = pts3d.mean(0)

    up = normalize(poses[:, :3, 1].mean(0))  # (3)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses[:, :3, 3] -= center
    poses_centered = R @ poses  # (N_images, 4, 4)

    if pts3d is not None:
        pts3d_centered = (pts3d - center) @ R[:3, :3].T
        # pts3d_centered = pts3d @ R[:3, :3].T - center
        return poses_centered, pts3d_centered

    return poses_centered


def visualize_poses(poses, size=0.05, bound=1, points=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2 * bound] * 3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2] * 3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array(
            [
                [pos, a],
                [pos, b],
                [pos, c],
                [pos, d],
                [a, b],
                [b, c],
                [c, d],
                [d, a],
                [pos, o],
            ]
        )
        segs = trimesh.load_path(segs)
        objects.append(segs)

    if points is not None:
        print(
            "[visualize points]",
            points.shape,
            points.dtype,
            points.min(0),
            points.max(0),
        )
        colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
        colors[:, 2] = 255  # blue
        colors[:, 3] = 30  # transparent
        objects.append(trimesh.PointCloud(points, colors))

    scene = trimesh.Scene(objects)
    scene.set_camera(distance=bound, center=[0, 0, 0])
    scene.show()
