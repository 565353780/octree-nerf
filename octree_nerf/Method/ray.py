import numpy as np
import torch

import trimesh
from octree_nerf.Method.mesh import custom_meshgrid


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None):
    """get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    """

    device = poses.device

    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = (
            intrinsics[:, 0],
            intrinsics[:, 1],
            intrinsics[:, 2],
            intrinsics[:, 3],
        )

    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    if N > 0:
        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1:
            # random sample left-top cores.
            num_patch = N // (patch_size**2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(
                torch.arange(patch_size, device=device),
                torch.arange(patch_size, device=device),
            )
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

        else:  # random sampling
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["i"] = i.long()
        results["j"] = j.long()

    else:
        inds = torch.arange(H * W, device=device)

    zs = -torch.ones_like(i)  # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy  # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1)  # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(
        1
    )  # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]

    rays_o = poses[:, :3, 3].expand_as(rays_d)  # [N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results


def visualize_rays(rays_o, rays_d):
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(0, rays_o.shape[0], 10):
        ro = rays_o[i]
        rd = rays_d[i]

        segs = np.array([[ro, ro + rd * 3]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()

    return True
