import torch


@torch.cuda.amp.autocast(enabled=False)
def proposal_loss(all_bins, all_weights):
    # all_bins: list of [N, T+1]
    # all_weights: list of [N, T]

    def loss_interlevel(t0, w0, t1, w1):
        # t0, t1: [N, T+1]
        # w0, w1: [N, T]
        cw1 = torch.cat(
            [torch.zeros_like(w1[..., :1]), torch.cumsum(w1, dim=-1)], dim=-1
        )
        inds_lo = (
            torch.searchsorted(
                t1[..., :-1].contiguous(), t0[..., :-1].contiguous(), right=True
            )
            - 1
        ).clamp(0, w1.shape[-1] - 1)
        inds_hi = torch.searchsorted(
            t1[..., 1:].contiguous(), t0[..., 1:].contiguous(), right=True
        ).clamp(0, w1.shape[-1] - 1)

        cw1_lo = torch.take_along_dim(cw1[..., :-1], inds_lo, dim=-1)
        cw1_hi = torch.take_along_dim(cw1[..., 1:], inds_hi, dim=-1)
        w = cw1_hi - cw1_lo

        return (w0 - w).clamp(min=0) ** 2 / (w0 + 1e-8)

    bins_ref = all_bins[-1].detach()
    weights_ref = all_weights[-1].detach()
    loss = 0
    for bins, weights in zip(all_bins[:-1], all_weights[:-1]):
        loss += loss_interlevel(bins_ref, weights_ref, bins, weights).mean()

    return loss
