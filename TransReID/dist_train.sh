# 
source /home/test/anaconda3/bin/activate SAMEnv
CUDA_VISIBLE_DEVICES=0,1  torchrun --nproc_per_node=2 --master_port 12345 train.py --config_file configs/Market/vit_base.yml MODEL.DIST_TRAIN True
# CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch  --nproc_per_node=2 --master_port 22222 train_hdr.py --config_file configs/Market/vit_hdr.yml MODEL.DIST_TRAIN True
# test
# python test.py --config_file configs/VehicleID/vit_base.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')"
