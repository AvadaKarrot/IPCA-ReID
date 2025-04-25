# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .l2loss import L2LOSS
from .aiaw_loss import AIAWLoss
from .caption_loss import CaptionLoss

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    ######################################### zwq for l2loss img_reconstruct - target_image 0602
    if cfg.HDRNET.HDR_NET:
         l2loss = L2LOSS()
        
    if cfg.MODEL.WHITE:
         aiawloss = AIAWLoss()

    if cfg.MODEL.CAPTION:
         caption_loss = CaptionLoss()

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'RandomSampler':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler != 'RandomSampler':
        def loss_func(score, feat, target, target_cam, sim=None,csa_output = None, img_reconstruct=None,target_image = None):
################ white loss @@@@@@@@@@@@@@@@@@@@@@@@@@@
            if cfg.MODEL.WHITE:
                AIAW_LOSS_1 = aiawloss(csa_output['feat_aug'], csa_output['eye'], csa_output['mask_matrix'],\
                                      csa_output['margin'], csa_output['num_remove_cov'])
                AIAW_LOSS_2 = aiawloss(csa_output['feat_org'], csa_output['eye'], csa_output['mask_matrix'],\
                                      csa_output['margin'], csa_output['num_remove_cov'])
            if cfg.MODEL.CAPTION:
                 ############ sim: (B,B)生成相似度矩阵，在makeloss，加入对比损失
                #  CAPTION_LOSS = caption_loss(sim) + caption_loss(sim.t()) / 2.0
                # CAPTION_LOSS =caption_loss(sim)
                CAPTION_LOSS = 0

            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS) 
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS) 
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS) 
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    # img reconstruction loss zwq 0602
                    if target_image is not None and img_reconstruct is not None:
                        L2LOSS = l2loss(img_reconstruct, target_image)
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                                cfg.MODEL.L2_LOSS_WEIGHT * L2LOSS
                    if cfg.MODEL.WHITE:
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +\
                               AIAW_LOSS_1 +  AIAW_LOSS_2 
                        # return AIAW_LOSS_1 +  AIAW_LOSS_2
                    if cfg.MODEL.CAPTION:
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +\
                               CAPTION_LOSS
                    
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


