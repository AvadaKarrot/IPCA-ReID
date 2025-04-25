import os
from config import cfg
import argparse
# from datasets import make_dataloader
from model.make_model_caption import make_model
from processor import do_inference
from utils.logger import setup_logger
from data import *
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from utils import CheckpointManager
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os.path as osp
import random
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/visual/clip_caption_coop.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

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

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    data_manager = build_datamanager(cfg)

    model = make_model(cfg, num_class=data_manager._num_train_pids, camera_num=data_manager._num_train_cams)
    model.load_param(cfg.TEST.WEIGHT)


    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(data_manager.num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    vid_list = []
    val_loader = data_manager.test_loader
    for n_iter, (img, vid, camid, camids, imgpath, captions) in enumerate(val_loader):
        # print(f'imgpath: {imgpath}')
        with torch.no_grad():
            img = img.to(device)
            # vid = vid.to(device)
            # camid = camid.to(device)
            camids = camids.to(device)
            # target_view = target_view.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids
            else: 
                camids = None
            feat = model(img, captions=captions,cam_label=camids)
            evaluator.update((feat, vid, camid))
            img_path_list.extend(imgpath)
            vid_list.extend(vid)
#############################################################################
    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    # distmat = evaluator.return_dist() # query*gallery
    # visualize_ranked_results(
    #     distmat,
    #     data_manager.fetch_test_loaders(cfg.DATASETS.TARGETS),
    #     data_manager.data_type,
    #     width=data_manager.width,
    #     height=data_manager.height,
    #     save_dir=osp.join(cfg.OUTPUT_DIR, 'visrank'+cfg.DATASETS.TARGETS),
    #     topk=20
    # )
##################################################################################
    pr_json_path = osp.join(cfg.OUTPUT_DIR, 'pr_curve_'+cfg.DATASETS.SOURCES + '2' +cfg.DATASETS.TARGETS+'.json')
    precision, recall = evaluator.computer_pr(pr_json_path) # query*gallery
    plt.plot(recall, precision, color='b', linestyle='-', linewidth=1)
    pr_curve_path = osp.join(cfg.OUTPUT_DIR, 'pr_curve_'+cfg.DATASETS.SOURCES + '2' +cfg.DATASETS.TARGETS+'.png')
    plt.savefig(pr_curve_path)
    # plt.show()