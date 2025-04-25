# train
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 22222 train_hdr.py --config_file configs/Market/vit_base.yml MODEL.DIST_TRAIN True
# CUDA_VISIBLE_DEVICES=0,1 /home/test/anaconda3/envs/SAMEnv/bin/torchrun --nproc_per_node=2 --master_port 11111 train_prompt.py --config_file configs/Market/vit_prompt.yml MODEL.DIST_TRAIN True
CUDA_VISIBLE_DEVICES=0,1 /home/test/anaconda3/envs/SAMEnv/bin/torchrun --nproc_per_node=2 --master_port 11112 train_caption.py --config_file configs/CUHK03/vit_caption_coop_adain.yml MODEL.DIST_TRAIN True

# test
# python test.py --config_file configs/Market/vit_clip.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
