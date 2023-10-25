import torch


# MeRF-like contraction
@torch.cuda.amp.autocast(enabled=False)
def contract(x):
    # x: [..., C]
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = x.abs().max(1, keepdim=True)  # [N, 1], [N, 1]
    scale = 1 / mag.repeat(1, C)
    scale.scatter_(1, idx, (2 - 1 / mag) / mag)
    z = torch.where(mag < 1, x, x * scale)
    return z.view(*shape, C)


@torch.cuda.amp.autocast(enabled=False)
def uncontract(z):
    # z: [..., C]
    shape, C = z.shape[:-1], z.shape[-1]
    z = z.view(-1, C)
    mag, idx = z.abs().max(1, keepdim=True)  # [N, 1], [N, 1]
    scale = 1 / (2 - mag.repeat(1, C)).clamp(min=1e-8)
    scale.scatter_(1, idx, 1 / (2 * mag - mag * mag).clamp(min=1e-8))
    x = torch.where(mag < 1, z, z * scale)
    return x.view(*shape, C)


@torch.cuda.amp.autocast(enabled=False)
def sample_pdf(bins, weights, T, perturb=False):
    # bins: [N, T0+1]
    # weights: [N, T0]
    # return: [N, T]

    N, T0 = weights.shape
    weights = weights + 0.01  # prevent NaNs
    weights_sum = torch.sum(weights, -1, keepdim=True)  # [N, 1]
    pdf = weights / weights_sum
    cdf = torch.cumsum(pdf, -1).clamp(max=1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [N, T+1]

    u = torch.linspace(0.5 / T, 1 - 0.5 / T, steps=T).to(weights.device)
    u = u.expand(N, T)

    if perturb:
        u = u + (torch.rand_like(u) - 0.5) / T

    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)  # [N, t]

    below = torch.clamp(inds - 1, 0, T0)
    above = torch.clamp(inds, 0, T0)

    cdf_g0 = torch.gather(cdf, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g0 = torch.gather(bins, -1, below)
    bins_g1 = torch.gather(bins, -1, above)

    bins_t = torch.clamp(
        torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0)), 0, 1
    )  # [N, t]
    bins = bins_g0 + bins_t * (bins_g1 - bins_g0)  # [N, t]

    return bins


@torch.cuda.amp.autocast(enabled=False)
def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.05):
    # rays: [N, 3], [N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [N, 1], far [N, 1]

    tmin = (aabb[:3] - rays_o) / (rays_d + 1e-15)  # [N, 3]
    tmax = (aabb[3:] - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).amax(dim=-1, keepdim=True)
    far = torch.where(tmin > tmax, tmin, tmax).amin(dim=-1, keepdim=True)
    # if far < near, means no intersection, set both near and far to inf (1e9 here)
    mask = far < near
    near[mask] = 1e9
    far[mask] = 1e9
    # restrict near to a minimal value
    near = torch.clamp(near, min=min_near)

    return near, far
