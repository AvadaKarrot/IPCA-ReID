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

from processor.validate_for_prototype import validate_for_prototype
from processor.validate_for_cov_stat import validate_for_cov_stat

def do_train(cfg,
             model,
             center_criterion,
             data_manager,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank):
    ######################################################
    train_loader = data_manager.train_loader
    val_loader = data_manager.test_loader
    ################## save best model ##################
    best_mAP = 0
    ################## save best model ##################
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    ############## 630 zwq fix 
    # device = torch.device("cuda")
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() >= 1 and cfg.MODEL.DIST_TRAIN: # zwq 701
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(data_manager.num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    manager = CheckpointManager(logs_dir = cfg.OUTPUT_DIR, model = model)

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        if cfg.MODEL.CSA:
            validate_for_prototype(cfg, train_loader, model, epoch, device, local_rank)
        if cfg.MODEL.WHITE:
            # model.cov_matrix_layer.reset_mask_matrix()
            model.module.cov_matrix_layer.reset_mask_matrix()
            validate_for_cov_stat(cfg, train_loader, model, epoch, device, local_rank)
            # model.cov_matrix_layer.set_mask_matrix()
            model.module.cov_matrix_layer.set_mask_matrix()
            for name, param in model.named_parameters():
                if 'cov_matrix_layer' in name:
                    param.requires_grad = True
                if 'csada_in' in name:
                    param.requires_grad = True           
                if 'final_conv' in name:
                    param.requires_grad = True    
        model.train()
        # print('ready to train')
        for n_iter, (img, vid, target_cam, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            # target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                if cfg.MODEL.WHITE:
                    score, feat, csa_output = model(img, target, cam_label=target_cam )
                    loss = loss_fn(score, feat, target, target_cam,csa_output)

                # original 
                else:
                    score, feat = model(img, target, cam_label=target_cam )
                    loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()
            if n_iter  == 100 and epoch == 1 :
                for name, param in model.named_parameters():
                    if 'text_encoder' not in name and param.grad is None:
                        print(f'Gradient of {name}: {param.grad}') 

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                #             .format(epoch, (n_iter + 1), len(train_loader),
                #                     loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
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
        ############## origin code for save model
        # if epoch % checkpoint_period == 0:
        #     if cfg.MODEL.DIST_TRAIN:
        #         if dist.get_rank() == 0:
        #             torch.save(model.state_dict(),
        #                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        #     else:
        #         torch.save(model.state_dict(),
        #                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, _, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            # vid = vid.to(device)
                            # camid = camid.to(device)
                            camids = camids.to(device)
                            # target_view = target_view.to(device)
                            feat = model(img, cam_label=camids)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    is_best = (mAP > best_mAP)
                    best_mAP = max(mAP, best_mAP)
                    if is_best:
                        manager.save(epoch=epoch, fpath=osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_epoch{}.pth'.format(epoch)))
                        manager.save_best_checkpoint(epoch=epoch, is_best=is_best, fpath=osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_epoch{}.pth'.format(epoch)))
                    # print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                    #         format(epoch, mAP, best_mAP, ' *' if is_best else ''))
                    logger.info('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                            format(epoch, mAP, best_mAP, ' *' if is_best else ''))
                    torch.cuda.empty_cache() 
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, _, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        # vid = vid.to(device)
                        # camid = camid.to(device)                        
                        camids = camids.to(device)
                        # target_view = target_view.to(device)
                        feat = model(img, cam_label=camids)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
                is_best = (mAP > best_mAP)
                best_mAP = max(mAP, best_mAP)
                if is_best:
                    manager.save(epoch=epoch, fpath=osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_epoch{}.pth'.format(epoch)))
                    manager.save_best_checkpoint(epoch=epoch, is_best=is_best, fpath=osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_epoch{}.pth'.format(epoch)))
                # print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                #         format(epoch, mAP, best_mAP, ' *' if is_best else ''))
                ################## zwq debug ################ log mAP best model
                logger.info('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                        format(epoch, mAP, best_mAP, ' *' if is_best else ''))
                
                torch.cuda.empty_cache()  
            # print('eval is end') 
            # dist.barrier()  
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

    for n_iter, (img, pid, camid, camids, imgpath ,target_view) in enumerate(val_loader):

        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            # target_view = target_view.to(device)
            feat = model(img, cam_label=None, view_label=None)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


