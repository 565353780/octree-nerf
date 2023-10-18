CUDA_VISIBLE_DEVICES=4 python main.py data/room/ \
	--workspace trial_ngp_360_room \
	--enable_cam_center \
	--enable_cam_near_far \
	--min_near 0.2 \
	--lambda_tv 0 \
	--lambda_distort 0 \
	--eval_cnt 1 \
	--save_cnt 1 \
	--downscale 4 \
	-O \
	--background random \
	--bound 8

