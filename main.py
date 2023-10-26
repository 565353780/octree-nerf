import torch
import numpy as np


from octree_nerf.Dataset.colmap import ColmapDataset as NeRFDataset
from octree_nerf.Config.config import getConfig
from octree_nerf.Metric.lpips import LPIPSMeter
from octree_nerf.Metric.psnr import PSNRMeter
from octree_nerf.Metric.ssim import SSIMMeter
from octree_nerf.Method.random_seed import seed_everything
from octree_nerf.Model.ngp import NeRFNetwork
from octree_nerf.Module.trainer import Trainer
from octree_nerf.Module.gui import NeRFGUI

# torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    opt = getConfig()

    if opt.O:
        opt.fp16 = True
        opt.preload = True
        opt.cuda_ray = True
        opt.mark_untrained = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True

    if opt.O2:
        opt.fp16 = True
        opt.bound = 128  # large enough
        opt.preload = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True

    seed_everything(opt.seed)

    model = NeRFNetwork(opt)

    # criterion = torch.nn.MSELoss(reduction='none')
    criterion = torch.nn.SmoothL1Loss(reduction="none")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.test:
        trainer = Trainer(
            opt,
            model,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            use_checkpoint=opt.ckpt,
        )

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            if not opt.test_no_video:
                test_loader = NeRFDataset(opt, device=device, type="test").dataloader()

                if test_loader.has_gt:
                    trainer.metrics = [
                        PSNRMeter(),
                        SSIMMeter(),
                        LPIPSMeter(device=device),
                    ]  # set up metrics
                    trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

                trainer.test(test_loader, write_video=True)  # test and save video

            # if not opt.test_no_mesh:
            #     # need train loader to get camera poses for visibility test
            #     if opt.mesh_visibility_culling:
            #         train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
            #     trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)

    else:
        optimizer = torch.optim.Adam(model.get_params(opt.lr), eps=1e-15)

        train_loader = NeRFDataset(
            opt, device=device, type=opt.train_split
        ).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(
            1, max_epoch // max(1, opt.save_cnt)
        )  # save ~50 times during the training
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(
            f"[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}."
        )

        # colmap can estimate a more compact AABB
        if opt.data_format == "colmap":
            model.update_aabb(train_loader._data.pts_aabb)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1)
        )

        trainer = Trainer(
            opt,
            model,
            workspace=opt.workspace,
            optimizer=optimizer,
            criterion=criterion,
            ema_decay=0.95,
            fp16=opt.fp16,
            lr_scheduler=scheduler,
            scheduler_update_every_step=True,
            use_checkpoint=opt.ckpt,
            eval_interval=eval_interval,
            save_interval=save_interval,
        )

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(opt, device=device, type="val").dataloader()

            trainer.metrics = [
                PSNRMeter(),
            ]
            trainer.train(train_loader, valid_loader, max_epoch)

            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
            trainer.evaluate(valid_loader)

            # also test
            test_loader = NeRFDataset(opt, device=device, type="test").dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader)  # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True)  # test and save video
            # trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
