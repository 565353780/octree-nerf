CUDA_VISIBLE_DEVICES=0 python main.py \
	/home/chli/Dataset/NeRF/3vjia_simple/ \
	--workspace 3vjia_simple_4000_OO2 \
	-O \
	-O2 \
	--enable_cam_center \
	--enable_cam_near_far \
	--min_near 0.2 \
	--lambda_tv 0 \
	--lambda_distort 0 \
	--eval_cnt 1 \
	--save_cnt 1 \
	--downscale 4 \
	--background random \
	--bound 8 \
	--num_rays 4000 \
	--max_ray_batch 16000
#--gui
