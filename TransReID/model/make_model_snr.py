import torch
import torch.nn as nn
from .backbones.resnet_snr import ResNet_snr, Bottleneck
import copy
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import numpy as np

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet_snr(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        # self.classifier1 = nn.Linear(256, self.num_classes,bias=False)
        # self.classifier2 = nn.Linear(512, self.num_classes,bias=False)
        # self.classifier3 = nn.Linear(1024, self.num_classes,bias=False)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        # self.classifier1.apply(weights_init_classifier)
        # self.classifier2.apply(weights_init_classifier)
        # self.classifier3.apply(weights_init_classifier)
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x_IN_1, x_1, x_1_useless,\
        x_IN_2, x_2, x_2_useless,\
        x_IN_3, x_3, x_3_useless,\
        x_IN_4, x_4, x_4_useless = self.base(x)

        x_1 = nn.functional.avg_pool2d(x_1, x_1.shape[2:4]) # b*256
        x_1 = x_1.view(x_1.shape[0], -1)  # flatten
        x_2 = nn.functional.avg_pool2d(x_2, x_2.shape[2:4]) # b*512
        x_2 = x_2.view(x_2.shape[0], -1)  # flatten 
        x_3 = nn.functional.avg_pool2d(x_3, x_3.shape[2:4]) # b*1024
        x_3 = x_3.view(x_3.shape[0], -1)  # flatten   
        
        x_IN_1 = nn.functional.avg_pool2d(x_IN_1, x_IN_1.shape[2:4])
        x_IN_1 = x_IN_1.view(x_IN_1.shape[0], -1)  # flatten
        x_IN_2 = nn.functional.avg_pool2d(x_IN_2, x_IN_2.shape[2:4])
        x_IN_2 = x_IN_2.view(x_IN_2.shape[0], -1)  # flatten 
        x_IN_3 = nn.functional.avg_pool2d(x_IN_3, x_IN_3.shape[2:4])
        x_IN_3 = x_IN_3.view(x_IN_3.shape[0], -1)  # flatten  
        x_IN_4 = nn.functional.avg_pool2d(x_IN_4, x_IN_4.shape[2:4])
        x_IN_4 = x_IN_4.view(x_IN_4.shape[0], -1)  # flatten  
                
        x_1_useless = nn.functional.avg_pool2d(x_1_useless, x_1_useless.shape[2:4])
        x_1_useless = x_1_useless.view(x_1_useless.shape[0], -1)  # flatten
        x_2_useless = nn.functional.avg_pool2d(x_2_useless, x_2_useless.shape[2:4])
        x_2_useless = x_2_useless.view(x_2_useless.shape[0], -1)  # flatten 
        x_3_useless = nn.functional.avg_pool2d(x_3_useless, x_3_useless.shape[2:4])
        x_3_useless = x_3_useless.view(x_3_useless.shape[0], -1)  # flatten          
        x_4_useless = nn.functional.avg_pool2d(x_4_useless, x_4_useless.shape[2:4])
        x_4_useless = x_4_useless.view(x_4_useless.shape[0], -1)  # flatten  
                
        global_feat = nn.functional.avg_pool2d(x_4, x_4.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat, x_IN_4, x_4_useless,\
                    x_3, x_IN_3, x_3_useless, \
                    x_2, x_IN_2, x_2_useless, x_1, x_IN_1, x_1_useless

        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        # param_dict = torch.load(trained_path)
        # if 'state_dict' in param_dict:
        #     param_dict = param_dict['state_dict']
        # for i in param_dict:
        #     self.state_dict()[i].copy_(param_dict[i])
        # print('Loading pretrained model from {}'.format(trained_path))
        param_dict = torch.load(trained_path)
        ######## debug #################
        param_dict = param_dict['model']
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))    
    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class):
    print('-------------build resnet srn --------------')
    return Backbone(num_class, cfg)