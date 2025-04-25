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
def print_gpu_memory_usage(step):
    print(f"{step} - Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    print(f"{step} - Cached:    {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(dist.get_world_size())]
    # print(torch.distributed.get_world_size())
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     rank = torch.distributed.get_rank()
#     world_size = torch.distributed.get_world_size()
#     print(f"Rank {rank}/{world_size}: Before all_gather, tensor size: {tensor.size()}")

#     tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
#     handle = torch.distributed.all_gather(tensors_gather, tensor, async_op=True)
#     handle.wait()  # 等待异步操作完成

#     print(f"Rank {rank}: After all_gather, gathered tensor sizes: {[t.size() for t in tensors_gather]}")

#     output = torch.cat(tensors_gather, dim=0)
#     print(f"Rank {rank}: After concatenation, output size: {output.size()}")
#     return output

@torch.no_grad()    
def compute_mean_std(
    feats: torch.Tensor, eps=1e-8, infer=False
) -> torch.Tensor:
    assert (
        len(feats.shape) == 4 or len(feats.shape) == 3
    ), "feature map should be 4-dimensional of the form N,C,H,W! or 3-dimensional of the form N,Seq,D!"
    #  * Doing this to support ONNX.js inference.
    # if infer:
    #     n = 1
    #     c = 512  # * fixed for vgg19
    # else:
    #     n, c, _, _ = feats.shape
    n, seq, c =feats.shape # b* 129 *768

    mean = torch.mean(feats[:, 1:, :], dim=-2).view(n, 1, c) # NO class token
    std = torch.std(feats[:, 1:, :], dim=-2).view(n, 1, c) + eps

    return mean, std

def validate_for_prototype(cfg, train_loader, model, epoch, device, local_rank):
    model.eval()
    style_list = torch.empty(size=(0, 2, 768)).cuda() # 0,2,C # 存储每个类别的style统计信息mean和std

    print('start select proto')
    for n_iter, (img, vid, target_cam, _, _) in enumerate(train_loader): # 遍历整个数据集
        img, vid, target_cam = img.to(device), vid.to(device), target_cam.to(device)
        with torch.no_grad():
            if cfg.MODEL.DIST_TRAIN:
                style = model.module.base(img, vid, target_cam) # b, 129, 768
            else:
                style = model.base(img, vid, target_cam) # b, 129, 768
            c_mean, c_std = compute_mean_std(style) # b,1, 768
            c_mean, c_std = c_mean.squeeze(1), c_std.squeeze(1) # b, 768
            statis = torch.stack([c_mean, c_std], dim=1) # b, 2, 768
            style_list = torch.cat((style_list, statis), dim=0)
            # print('{} iter has {} image'.format(n_iter, img.size(0))) # 检查dataloaderiter的次数

            del img, vid, target_cam, c_mean, c_std, statis
            ######## loging
            if local_rank == 0:
                if n_iter % 50 == 0:
                    print('select proto idx:', n_iter)
    # if cfg.MODEL.DIST_TRAIN:
    #     style_list = concat_all_gather(style_list)# N,2,768
        # print('after concat {}'.format(style_list.device))
    style_list = style_list.reshape(style_list.size(0), -1).detach()
    # print('after detach {}'.format(style_list.device))
    proto_style,_ = farthest_point_sample_tensor(style_list, cfg.MODEL.CSA_STYLE_NUM) # L, 2, 768
    # print('after fps {}'.format(proto_style.device))
    proto_style = proto_style.reshape(cfg.MODEL.CSA_STYLE_NUM, 2, 768)
    proto_mean, proto_std = proto_style[:,0], proto_style[:,1] # L, 768
    if cfg.MODEL.DIST_TRAIN:
        model.module.csada_in.proto_mean.copy_(proto_mean)
        model.module.csada_in.proto_std.copy_(proto_std)
    else:
        model.csada_in.proto_mean.copy_(proto_mean)
        model.csada_in.proto_std.copy_(proto_std)    
    del style_list, proto_style, proto_mean, proto_std
    # model.train() # in the process.py
    ########### 再进行这一轮EPOCH的模型训练


