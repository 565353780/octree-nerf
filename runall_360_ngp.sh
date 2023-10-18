CUDA_VISIBLE_DEVICES=4 python main.py \
	/home/chli/Dataset/NeRF/3vjia_simple/ \
	--workspace logs/3vjia_simple \
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
