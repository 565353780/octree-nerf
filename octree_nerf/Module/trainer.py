import glob
import os
import time

import cv2
import imageio
import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from octree_nerf.Config.config import getConfig
from octree_nerf.Method.ray import get_rays
from octree_nerf.Method.random_seed import seed_everything
from octree_nerf.Dataset.colmap import ColmapDataset
from octree_nerf.Metric.lpips import LPIPSMeter
from octree_nerf.Metric.psnr import PSNRMeter
from octree_nerf.Metric.ssim import SSIMMeter
from octree_nerf.Model.ngp import NeRFNetwork

class Trainer(object):
    def __init__(
        self,
    ):
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.console = Console()
        self.log_ptr = None

        self.opt = getConfig()

        if self.opt.O:
            self.opt.fp16 = True
            self.opt.preload = True
            self.opt.adaptive_num_rays = True
            self.opt.random_image_batch = True
            self.opt.cuda_ray = True
            self.opt.mark_untrained = True

        if self.opt.O2:
            self.opt.fp16 = True
            self.opt.preload = True
            self.opt.adaptive_num_rays = True
            self.opt.random_image_batch = True
            self.opt.bound = 128  # large enough

        seed_everything(self.opt.seed)

        self.scheduler_update_every_step=False
        self.report_metric_at_train = True
        self.use_checkpoint = 'latest'
        self.batch_size = 1
        self.ema_decay = 0.95
        self.max_keep_ckpt = 2
        self.eval_interval = 100
        self.save_interval = 100

        self.workspace = self.opt.workspace
        os.makedirs('./logs/' + self.workspace + '/', exist_ok=True)
        os.makedirs('./output/' + self.workspace + '/', exist_ok=True)

        self.fp16 = self.opt.fp16

        # self.criterion = torch.nn.MSELoss(reduction='none').to(self.device)
        self.criterion = torch.nn.SmoothL1Loss(reduction="none").to(self.device)

        self.train_loader = ColmapDataset(self.opt, device=self.device, type=self.opt.train_split).dataloader(self.batch_size)
        self.test_loader = ColmapDataset(self.opt, device=self.device, type="test").dataloader(self.batch_size)
        self.valid_loader = ColmapDataset(self.opt, device=self.device, type="val").dataloader(self.batch_size)

        self.metrics = [PSNRMeter()]
        if self.test_loader.has_gt:
            self.metrics.append(SSIMMeter())
            self.metrics.append(LPIPSMeter(device=self.device))

        self.model = NeRFNetwork(self.opt).to(self.device)
        self.model.update_aabb(self.train_loader._data.pts_aabb)
        if torch.__version__[0] == "2":
            self.model = torch.compile(self.model)

        self.optimizer = torch.optim.Adam(self.model.get_params(self.opt.lr * self.batch_size), eps=1e-15)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda iter: 0.1 ** min(iter / self.opt.iters * self.batch_size, 1)
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            self.log_path = './output/' + self.workspace + "/log.txt"
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = './output/' + self.workspace + "/checkpoints"
            self.best_path = f"{self.ckpt_path}/model_best.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.workspace} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"}'
        )
        self.log(
            f"[INFO] #parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}"
        )

        self.log(self.opt)
        self.log(self.model)

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        rays_o = data["rays_o"].reshape(-1, 3)  # [BxN, 3]
        rays_d = data["rays_d"].reshape(-1, 3)  # [BxN, 3]
        index = data["index"].reshape(-1)  # [BxN]
        cam_near_far = (
            data["cam_near_far"] if "cam_near_far" in data else None
        ).reshape(-1, 2)  # [BxN, 2] or None

        images = data["images"].reshape(-1, data["images"].shape[2])  # [BxN, 3/4]

        N, C = images.shape

        if self.opt.background == "random":
            bg_color = torch.rand(
                N, 3, device=self.device
            )  # [BxN, 3], pixel-wise random.
        else:  # white / last_sample
            bg_color = 1

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                1 - images[..., 3:]
            )
        else:
            gt_rgb = images

        shading = "diffuse" if self.global_step < self.opt.diffuse_step else "full"
        update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0

        outputs = self.model.render(
            rays_o,
            rays_d,
            index=index,
            bg_color=bg_color,
            perturb=True,
            cam_near_far=cam_near_far,
            shading=shading,
            update_proposal=update_proposal,
        )

        # MSE loss
        pred_rgb = outputs["image"]
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)  # [BxN, 3] --> [BxN]

        loss = loss.mean()

        # extra loss
        if "proposal_loss" in outputs and self.opt.lambda_proposal > 0:
            loss = loss + self.opt.lambda_proposal * outputs["proposal_loss"]

        if "distort_loss" in outputs and self.opt.lambda_distort > 0:
            loss = loss + self.opt.lambda_distort * outputs["distort_loss"]

        if self.opt.lambda_entropy > 0:
            w = outputs["weights_sum"].clamp(1e-5, 1 - 1e-5)
            entropy = -w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
            loss = loss + self.opt.lambda_entropy * (entropy.mean())

        # adaptive num_rays
        if self.opt.adaptive_num_rays:
            self.opt.num_rays = int(
                round((self.opt.num_points / outputs["num_points"]) * self.opt.num_rays)
            )

        return pred_rgb, gt_rgb, loss

    def post_train_step(self):
        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            self.model.apply_total_variation(self.opt.lambda_tv)

        if self.opt.lambda_wd > 0:
            self.model.apply_weight_decay(self.opt.lambda_wd)

    def eval_step(self, data):
        rays_o = data["rays_o"]  # [N, 3]
        rays_d = data["rays_d"]  # [N, 3]
        images = data["images"]  # [H, W, 3/4]
        index = data["index"]  # [1/N]
        H, W, C = images.shape

        cam_near_far = (
            data["cam_near_far"] if "cam_near_far" in data else None
        )  # [1/N, 2] or None

        # eval with fixed white background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (
                1 - images[..., 3:]
            )
        else:
            gt_rgb = images

        outputs = self.model.render(
            rays_o,
            rays_d,
            index=index,
            bg_color=bg_color,
            perturb=False,
            cam_near_far=cam_near_far,
        )

        pred_rgb = outputs["image"].reshape(H, W, 3)
        pred_depth = outputs["depth"].reshape(H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, shading="full"):
        rays_o = data["rays_o"]  # [N, 3]
        rays_d = data["rays_d"]  # [N, 3]
        index = data["index"]  # [1/N]
        H, W = data["H"], data["W"]

        cam_near_far = (
            data["cam_near_far"] if "cam_near_far" in data else None
        )  # [1/N, 2] or None

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(
            rays_o,
            rays_d,
            index=index,
            bg_color=bg_color,
            perturb=perturb,
            cam_near_far=cam_near_far,
            shading=shading,
        )

        pred_rgb = outputs["image"].reshape(H, W, 3)
        pred_depth = outputs["depth"].reshape(H, W)

        return pred_rgb, pred_depth

    def save_mesh(
        self, save_path=None, resolution=128, decimate_target=1e5, dataset=None
    ):
        if save_path is None:
            save_path = './output/' + self.workspace + "/mesh/"

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(
            save_path,
            resolution=resolution,
            decimate_target=decimate_target,
            dataset=dataset,
        )

        self.log("==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        self.writer = tensorboardX.SummaryWriter('./logs/' + self.workspace)

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if (
                (self.epoch % self.save_interval == 0 or self.epoch == max_epochs)
                    and self.workspace is not None
            ):
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.6f} minutes.")

        self.writer.close()

    def evaluate(self, loader, name=None):
        self.evaluate_one_epoch(loader, name)

    def test(self, loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = './output/' + self.workspace + "/results/"

        if name is None:
            name = f"ngp_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                preds, preds_depth = self.test_step(data)
                pred = preds.detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth.detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (
                    pred_depth.max() - pred_depth.min() + 1e-6
                )
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(
                        os.path.join(save_path, f"{name}_{i:04d}_rgb.png"),
                        cv2.cvtColor(pred, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(
                        os.path.join(save_path, f"{name}_{i:04d}_depth.png"), pred_depth
                    )

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)  # [N, H, W, 3]
            all_preds_depth = np.stack(all_preds_depth, axis=0)  # [N, H, W]

            # fix ffmpeg not divisible by 2
            all_preds = np.pad(
                all_preds,
                (
                    (0, 0),
                    (0, 1 if all_preds.shape[1] % 2 != 0 else 0),
                    (0, 1 if all_preds.shape[2] % 2 != 0 else 0),
                    (0, 0),
                ),
            )
            all_preds_depth = np.pad(
                all_preds_depth,
                (
                    (0, 0),
                    (0, 1 if all_preds_depth.shape[1] % 2 != 0 else 0),
                    (0, 1 if all_preds_depth.shape[2] % 2 != 0 else 0),
                ),
            )

            imageio.mimwrite(
                os.path.join(save_path, f"{name}_rgb.mp4"),
                all_preds,
                fps=24,
                quality=8,
                macro_block_size=1,
            )
            imageio.mimwrite(
                os.path.join(save_path, f"{name}_depth.mp4"),
                all_preds_depth,
                fps=24,
                quality=8,
                macro_block_size=1,
            )

        self.log("==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):
        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0 and self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)

        for _ in range(step):
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if (
                self.model.cuda_ray
                and self.global_step % self.opt.update_extra_interval == 0
            ):
                self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            preds, truths, loss_net = self.train_step(data)

            loss = loss_net

            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss_net.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            "loss": average_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return outputs

    # [GUI] test on a single image
    def test_gui(
        self,
        pose,
        intrinsics,
        mvp,
        W,
        H,
        bg_color=None,
        spp=1,
        downscale=1,
        shading="full",
    ):
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            "mvp": mvp,
            "rays_o": rays["rays_o"],
            "rays_d": rays["rays_d"],
            "H": rH,
            "W": rW,
            "index": [0],
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            # here spp is used as perturb random seed! (but not perturb the first sample)
            preds, preds_depth = self.test_step(
                data,
                bg_color=bg_color,
                perturb=False if spp == 1 else spp,
                shading=shading,
            )

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = (
                F.interpolate(
                    preds.unsqueeze(0).permute(0, 3, 1, 2), size=(H, W), mode="nearest"
                )
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .contiguous()
            )
            preds_depth = (
                F.interpolate(
                    preds_depth.unsqueeze(0).unsqueeze(1), size=(H, W), mode="nearest"
                )
                .squeeze(0)
                .squeeze(1)
            )

        pred = preds.detach().cpu().numpy()
        pred_depth = preds_depth.detach().cpu().numpy()

        outputs = {
            "image": pred,
            "depth": pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ..."
        )

        total_loss = 0
        if self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        self.local_step = 0

        for data in loader:
            if False:
                print('data:')
                for key, item in data.items():
                    if isinstance(item, torch.Tensor):
                        print(key, '-->', item.shape)

            # update grid every 16 steps
            if (
                self.model.cuda_ray
                and self.global_step % self.opt.update_extra_interval == 0
            ):
                self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            preds, truths, loss_net = self.train_step(data)

            loss = loss_net

            self.scaler.scale(loss).backward()

            self.post_train_step()  # for TV loss...

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss_net.item()
            total_loss += loss_val

            if self.report_metric_at_train:
                for metric in self.metrics:
                    metric.update(preds, truths)

            self.writer.add_scalar("train/loss", loss_val, self.global_step)
            self.writer.add_scalar(
                "train/lr",
                self.optimizer.param_groups[0]["lr"],
                self.global_step,
            )

            if self.scheduler_update_every_step:
                pbar.set_description(
                    f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                pbar.set_description(
                    f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})"
                )
            pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        pbar.close()
        if self.report_metric_at_train:
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        total_loss = 0
        for metric in self.metrics:
            metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                preds, preds_depth, truths, loss = self.eval_step(data)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                metric_vals = []
                for metric in self.metrics:
                    metric_val = metric.update(preds, truths)
                    metric_vals.append(metric_val)

                # save image
                save_path = './output/' + self.workspace + "/validation/" + \
                    f"{name}_{self.local_step:04d}_rgb.png"

                save_path_depth = './output/' + self.workspace + "/validation/" + \
                    f"{name}_{self.local_step:04d}_depth.png"

                save_path_error = './output/' + self.workspace + "/validation/" + \
                    f"{name}_{self.local_step:04d}_error_{metric_vals[0]:.2f}.png"

                # self.log(f"==> Saving validation image to {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                pred = preds.detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth.detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (
                    pred_depth.max() - pred_depth.min() + 1e-6
                )
                pred_depth = (pred_depth * 255).astype(np.uint8)

                truth = truths.detach().cpu().numpy()
                truth = (truth * 255).astype(np.uint8)
                error = (
                    np.abs(truth.astype(np.float32) - pred.astype(np.float32))
                    .mean(-1)
                    .astype(np.uint8)
                )

                cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_path_depth, pred_depth)
                cv2.imwrite(save_path_error, error)

                pbar.set_description(
                    f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})"
                )
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        pbar.close()
        self.stats["results"].append(average_loss)

        for metric in self.metrics:
            self.log(metric.report(), style="blue")
            metric.write(self.writer, self.epoch, prefix="evaluate")
            metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if self.model.cuda_ray:
            state["mean_density"] = self.model.mean_density

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:
            state["model"] = self.model.state_dict()

            file_path = f"{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(
                        self.ckpt_path, self.stats["checkpoints"].pop(0)
                    )
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                if (
                    self.stats["best_result"] is None
                    or self.stats["results"][-1] < self.stats["best_result"]
                ):
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}"
                    )
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    "[WARN] no evaluated results found, skip saving best checkpoint."
                )

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:  # load latest
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/*.pth"))

            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, abort loading latest model.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict["ema"])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if "mean_density" in checkpoint_dict:
                self.model.mean_density = checkpoint_dict["mean_density"]

        if model_only:
            return

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
