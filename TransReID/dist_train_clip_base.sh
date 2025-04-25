# train
CUDA_VISIBLE_DEVICES=0,1 /home/test/anaconda3/envs/SAMEnv/bin/torchrun --nproc_per_node=2 --master_port 11112 train_clipreid_base.py --config_file configs/CUHK03/vit_clip_base.yml MODEL.DIST_TRAIN True

# test
# python test.py --config_file configs/Market/vit_clip.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
