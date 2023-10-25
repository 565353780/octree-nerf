import os
import torch
import lpips


class LPIPSMeter:
    def __init__(self, net="vgg", device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for inp in inputs:
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(
            truths, preds, normalize=True
        ).item()  # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

        return v

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(
            os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step
        )

    def report(self):
        return f"LPIPS ({self.net}) = {self.measure():.6f}"
