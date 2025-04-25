import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from utils import CheckpointManager
from utils.fps import farthest_point_sample_tensor
import os.path as osp
import torch.autograd.profiler as profiler
from data.sampler_ddp import all_gather

def validate_for_cov_stat(cfg, train_loader, model, epoch, device, local_rank):
    model.eval()

    print('start caculate cov stat')
    for n_iter, (img, vid, target_cam, _, _) in enumerate(train_loader): # 遍历整个数据集
        img, vid, target_cam = img.to(device), vid.to(device), target_cam.to(device)
        with torch.no_grad():
            model(img, vid, target_cam, cal_covstat=True) # b, 129, 768# Forward pass
        
        del img, vid, target_cam

        # Logging
        if local_rank == 0:
            if n_iter % 100 == 0:
                print('update mask idx:', n_iter)
                # break
            # if idx > 10:
            #     break