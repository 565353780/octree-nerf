import torch
from torch_efficient_distloss import eff_distloss


@torch.cuda.amp.autocast(enabled=False)
def distort_loss(bins, weights):
    # bins: [N, T+1], in [0, 1]
    # weights: [N, T]

    intervals = bins[..., 1:] - bins[..., :-1]
    mid_points = bins[..., :-1] + intervals / 2

    loss = eff_distloss(weights, mid_points, intervals)

    return loss
