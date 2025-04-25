from utils.logger import setup_logger
# from datasets import make_dataloader
from model.make_model_clipreid_base import make_model
from solver.make_optimizer_clipreid import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss.make_loss_clipreid import make_loss
from processor.processor_clipreid_base import do_train_clipreid_base
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg

from data import *

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
        "--config_file", default="configs/MSMT17/vit_clip_base.yml", help="path to config file", type=str
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
    
    ####################zwq debug####################
    local_rank = int(os.getenv('LOCAL_RANK', 0))

    if cfg.MODEL.DIST_TRAIN:
        # torch.cuda.set_device(args.local_rank) # zwq
        torch.cuda.set_device(local_rank)
        

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    print(f'The Prompt: {cfg.MODEL.PROMPT}')
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

    ######### view_num = 0 default ###################
    model = make_model(cfg, num_class=data_manager._num_train_pids, camera_num=data_manager._num_train_cams, view_num=0)

    loss_func, center_criterion = make_loss(cfg, num_classes=data_manager._num_train_pids)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)


    do_train_clipreid_base(
        cfg,
        model,
        center_criterion,
        data_manager,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        local_rank
    )
