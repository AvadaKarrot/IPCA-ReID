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

import os.path as osp
import re

def do_pretrain(cfg,
             model,
             data_manager,
             optimizer,
             scheduler,
             loss_fn,
             local_rank):
    ######################################################
    train_loader = data_manager.train_loader
    val_loader = data_manager.test_loader

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    resume = 0 ############ zwq: 如果想要继续训练，加载test weight
    if cfg.SOLVER.RESUME_TRAIN:
        match = re.search(r'epoch(\d+)', cfg.TEST.WEIGHT)
        resume = int(match.group(1))

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()


    scaler = amp.GradScaler()
    manager = CheckpointManager(logs_dir = cfg.OUTPUT_DIR, model = model)

    # train
    for epoch in range(1+resume, epochs + 1 +resume):
        start_time = time.time()
        loss_meter.reset()

        scheduler.step(epoch)
        model.train()

        for n_iter, (img, low, full, target, _, _, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            low = low.to(device)
            full = full.to(device)
            target = target.to(device)
            # target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                img_reconstruct, _  = model(low, full )
                loss = loss_fn(img_reconstruct, target)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])


            torch.cuda.synchronize()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    #############3 zwq #############
                    manager.save(epoch=epoch, fpath=osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_epoch{}.pth'.format(epoch)))
                    # torch.save(model.state_dict(),
                    #            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                # torch.save(model.state_dict(),
                #            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                manager.save(epoch=epoch, fpath=osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_epoch{}.pth'.format(epoch)))




def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


