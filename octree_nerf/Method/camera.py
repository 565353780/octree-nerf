import numpy as np


def create_dodecahedron_cameras(radius=1, center=np.array([0, 0, 0])):
    vertices = np.array(
        [
            -0.57735,
            -0.57735,
            0.57735,
            0.934172,
            0.356822,
            0,
            0.934172,
            -0.356822,
            0,
            -0.934172,
            0.356822,
            0,
            -0.934172,
            -0.356822,
            0,
            0,
            0.934172,
            0.356822,
            0,
            0.934172,
            -0.356822,
            0.356822,
            0,
            -0.934172,
            -0.356822,
            0,
            -0.934172,
            0,
            -0.934172,
            -0.356822,
            0,
            -0.934172,
            0.356822,
            0.356822,
            0,
            0.934172,
            -0.356822,
            0,
            0.934172,
            0.57735,
            0.57735,
            -0.57735,
            0.57735,
            0.57735,
            0.57735,
            -0.57735,
            0.57735,
            -0.57735,
            -0.57735,
            0.57735,
            0.57735,
            0.57735,
            -0.57735,
            -0.57735,
            0.57735,
            -0.57735,
            0.57735,
            -0.57735,
            -0.57735,
            -0.57735,
        ]
    ).reshape((-1, 3), order="C")

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    # construct camera poses by lookat
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # forward is simple, notice that it is in fact the inversion of camera direction!
    forward_vector = normalize(vertices - center)
    # pick a temp up_vector, usually [0, 1, 0]
    up_vector = np.array([0, 1, 0], dtype=np.float32)[None].repeat(
        forward_vector.shape[0], 0
    )
    # cross(up, forward) --> right
    right_vector = normalize(np.cross(up_vector, forward_vector, axis=-1))
    # rectify up_vector, by cross(forward, right) --> up
    up_vector = normalize(np.cross(forward_vector, right_vector, axis=-1))

    ### construct c2w
    poses = np.eye(4, dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)
    poses[:, :3, 3] = vertices

    return poses
