import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
######################################################################## 
        # 冻结 HDRPointwiseNN 中除了 self.coeffs 的其他参数
        if 'hdrnet' in key and 'coeffs' not in key:
            value.requires_grad = False        
######################################################################## zwq 0602
        if 'text_encoder' in key:
            value.requires_grad = False # 不进行文本特征提取学习 冻结参数
        if 'image_encoder' in key:
            if key == 'image_encoder.positional_embedding':
                value.requires_grad = False
            else:
                value.requires_grad = False # 不进行image特征提取学习 冻结参数

        if 'image_projection' in key:
            value.requires_grad = True
        if 'text_projection' in key:
            value.requires_grad = True
        if 'text_encoder.text_projection' in key:
            value.requires_grad = False        
        if not value.requires_grad:
            continue
        
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    #zwq's test
    for k, v in model.named_parameters():
        if v.requires_grad == True:
            print(f'param: {k}')
    return optimizer, optimizer_center
