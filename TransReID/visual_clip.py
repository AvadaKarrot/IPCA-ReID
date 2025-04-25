import torch
import numpy as np
import os
import argparse
import random
from config import cfg
from utils.logger import setup_logger
import torch.nn as nn
import os
import os.path as osp

from PIL import Image
import cv2

import time
# from utils.Visualizer.visualizer import get_local
# from utils.Visualizer.visualizer import *
# get_local.activate() # 激活装饰器

# from model.make_model_caption import make_model
from model.make_model_caption import make_model
from data import *

from utils.pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from utils.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utils.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def reshape_transform(tensor, height=16, width=8):
    result = tensor[1:, :, :].reshape(height, width, tensor.size(1), tensor.size(2)) #16,8,bs,dim
    # result = tensor[:, 1:, :].reshape(height, width, tensor.size(0), tensor.size(2)) #16,8,bs,dim
    result = result.permute(2,3,0,1) #     # Bring the channels to the first dimension,
    return result # bs dim h*w

def imagedata_kwargs(cfg):
    return {
        'root': cfg.DATASETS.ROOT_DIR,
        'sources': cfg.DATASETS.SOURCES,
        'targets': cfg.DATASETS.TARGETS,
        'height': cfg.INPUT.SIZE_TRAIN[0],
        'width': cfg.INPUT.SIZE_TRAIN[1],
        'transforms': cfg.DATASETS.TRANSFORMS,
        'norm_mean': cfg.INPUT.PIXEL_MEAN,
        'norm_std': cfg.INPUT.PIXEL_STD,
        'batch_size_train': cfg.SOLVER.IMS_PER_BATCH ,
        'batch_size_test': cfg.TEST.IMS_PER_BATCH ,
        'workers': cfg.DATALOADER.NUM_WORKERS,
        'num_instances': cfg.DATALOADER.NUM_INSTANCE,
        'train_sampler': cfg.DATALOADER.SAMPLER,
        'dist_train': cfg.MODEL.DIST_TRAIN,
        'randomerase_prob': cfg.INPUT.RE_PROB,
        'padding' : cfg.INPUT.PADDING,
        'sobel_prob': cfg.INPUT.SOBEL_PROB,
        'hdrnet': cfg.HDRNET.HDR_NET,
        'caption': cfg.MODEL.CAPTION,
        'cap_num': cfg.CAPTION.CAP_NUM
        # image dataset specific
    }

def build_datamanager(cfg):
    return ImageDataManager(
        **imagedata_kwargs(cfg)
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/visual/clip_caption.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()    
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    set_seed(cfg.SOLVER.SEED)
    
    local_rank = int(os.getenv('LOCAL_RANK', 0))

    if cfg.MODEL.DIST_TRAIN:
        # torch.cuda.set_device(args.local_rank) # zwq
        torch.cuda.set_device(local_rank)
        
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    data_manager = build_datamanager(cfg)
    
    gallery_loader = data_manager.train_loader_normal
    train_normal = data_manager.trainset_normal

    model = make_model(cfg, num_class=data_manager._num_train_pids, camera_num=data_manager._num_train_cams)
    
    # device = 'cuda'
    device = torch.device('cuda:0')
    model.load_param(cfg.TEST.WEIGHT)

    if device:
        # if torch.cuda.device_count() > 1:
        #     print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        #     model = nn.DataParallel(model)
        model.to(device)
    target_layers = [model.image_encoder.transformer.resblocks[i].ln_1 for i in range(9, 12)]
    cam = GradCAM(model=model, 
                  target_layers = target_layers,
                  reshape_transform = reshape_transform)       
    start_time = time.time()    
    for n_iter, (img, _, _, _, img_path, captions) in enumerate(gallery_loader):
        targets = None
        grayscale_cam = cam(img, captions, targets=targets)

        for i in range(len(grayscale_cam)):

            rbg_img = cv2.imread(img_path[i]) # 用文件路径访问
            rgb_img  = cv2.resize(rbg_img, (128, 256))
            rgb_img = np.float32(rgb_img) / 255
            save_path = osp.join(cfg.OUTPUT_DIR, osp.basename(img_path[i]))

            grayscale_cam_i = grayscale_cam[i, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam_i)
            cv2.imwrite(save_path, cam_image)
    end_time = time.time()
    logger.info('Time: {}'.format(end_time - start_time))        
