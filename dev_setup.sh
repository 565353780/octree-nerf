pip install rich tqdm ninja torch numpy scipy \
	lpips pandas trimesh PyMCubes torch-ema \
	dearpygui packaging matplotlib tensorboardX \
	opencv-python imageio imageio-ffmpeg pymeshlab \
	xatlas scikit-learn torchmetrics tensorboard

# torch-scatter
pip install torch_efficient_distloss
pip install git+https://github.com/NVlabs/nvdiffrast/

cd ./octree_nerf/Lib/
pip install ./raymarching
pip install ./gridencoder
pip install ./freqencoder
pip install ./shencoder
cd ../..
