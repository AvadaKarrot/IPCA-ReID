from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil

import torch
import torch.nn as nn
from torch.nn import Parameter

from .osutils import mkdir_if_missing

# copy from spcl
def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)

def save_best_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))

def load_checkpoint(fpath):
    assert osp.isdir(fpath) or osp.isfile(fpath), 'previous checkpoint path not exists or not a folder'
    fpath = osp.join(fpath, 'checkpoint.pth.tar') if osp.isdir(fpath) else fpath

    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


class CheckpointManager(object):
    def __init__(self, logs_dir='./logs', **modules):
        self.logs_dir = logs_dir
        self.modules = modules

    def save(self, epoch, fpath=None, **modules):
        ckpt = {}
        modules.update(self.modules)
        for name, module in modules.items():
            if isinstance(module, nn.DataParallel):
                ckpt[name] = module.module.state_dict()
            else:
                ckpt[name] = module.state_dict()
        ckpt['epoch'] = epoch + 1

        fpath = osp.join(self.logs_dir, f"checkpoint-epoch{epoch}.pth.tar") if fpath is None else fpath
        save_checkpoint(ckpt, fpath)
    def save_PS_KD(self, epoch, Past_predictions, fpath=None, **modules):
        ckpt = {}
        modules.update(self.modules)
        for name, module in modules.items():
            if isinstance(module, nn.DataParallel):
                ckpt[name] = module.module.state_dict()
            else:
                ckpt[name] = module.state_dict()
        ckpt['epoch'] = epoch + 1
        ckpt['Past_predictions'] = Past_predictions
        fpath = osp.join(self.logs_dir, f"checkpoint-epoch{epoch}.pth.tar") if fpath is None else fpath
        save_checkpoint(ckpt, fpath)   
        
    def save_best_checkpoint(self, epoch, is_best, fpath=None, **modules):
        ckpt = {}
        for name, module in modules.items():
            if isinstance(module, nn.DataParallel):
                ckpt[name] = module.module.state_dict()
            else:
                ckpt[name] = module.state_dict()
        ckpt['epoch'] = epoch + 1
        fpath = osp.join(self.logs_dir, f"checkpoint-epoch{epoch}.pth.tar") if fpath is None else fpath        
        # mkdir_if_missing(osp.dirname(fpath))
        if is_best:
            shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))

    def load(self, ckpt):
        for name, module in self.modules.items():
            missing_keys, unexpected_keys = module.load_state_dict(ckpt.get(name, {}), strict=False)
            print(f"Loading {name}... \n"
                  f"missing keys {missing_keys} \n"
                  f"unexpected keys {unexpected_keys} \n")
        return ckpt["epoch"]
