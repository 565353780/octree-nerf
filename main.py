from octree_nerf.Module.trainer import Trainer
from octree_nerf.Module.gui import NeRFGUI

if __name__ == "__main__":
    trainer = Trainer()

    if trainer.opt.test:
        if trainer.opt.gui:
            gui = NeRFGUI(trainer.opt, trainer)
            gui.render()

        elif not trainer.opt.test_no_video:
            if trainer.test_loader.has_gt:
                trainer.evaluate(trainer.test_loader)  # blender has gt, so evaluate it.

            trainer.test(trainer.test_loader, write_video=True)  # test and save video

    else:
        if trainer.opt.gui:
            gui = NeRFGUI(trainer.opt, trainer, trainer.train_loader)
            gui.render()

        else:
            trainer.train(trainer.train_loader, trainer.valid_loader, 1000)

            # last validation
            trainer.evaluate(trainer.valid_loader)

            # also test
            if trainer.test_loader.has_gt:
                trainer.evaluate(trainer.test_loader)  # blender has gt, so evaluate it.

            trainer.test(trainer.test_loader, write_video=True)  # test and save video
            # trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
