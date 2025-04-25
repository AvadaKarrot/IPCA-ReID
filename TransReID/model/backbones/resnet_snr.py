import math

import torch
from torch import nn

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x # R
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x # b*256*1*1
        return input * x, input * (1 - x), x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_snr(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck,layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        # IN bridge:
        self.IN1 = nn.InstanceNorm2d(256, affine=True)
        self.IN2 = nn.InstanceNorm2d(512, affine=True)
        self.IN3 = nn.InstanceNorm2d(1024, affine=True)
        self.IN4 = nn.InstanceNorm2d(2048, affine=True)
        
        # SE for selection:
        self.style_reid_laye1 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye2 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye3 = ChannelGate_sub(1024, num_gates=1024, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)
        self.style_reid_laye4 = ChannelGate_sub(2048, num_gates=2048, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)   
             
    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, cam_label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x) # b * 64 64 32
        
        x_1 = self.layer1(x) # b * 256, 64, 32
        x_1_org = x_1
        x_IN_1 = self.IN1(x_1) # b * 256, 64, 32
        x_style_1 = x_1 - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_laye1(x_style_1)
        x_1 = x_IN_1 + x_style_1_reid_useful  # b * 256, 64, 32
        x_1_useless = x_IN_1 + x_style_1_reid_useless
        
        x_2 = self.layer2(x_1)  # torch.Size([64, 512, 32, 16])
        x_2_ori = x_2
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2_reid_useful, x_style_2_reid_useless, selective_weight_useful_2 = self.style_reid_laye2(x_style_2)
        x_2 = x_IN_2 + x_style_2_reid_useful
        x_2_useless = x_IN_2 + x_style_2_reid_useless

        x_3 = self.layer3(x_2)  # torch.Size([64, 1024, 16, 8])
        x_3_ori = x_3
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3_reid_useful, x_style_3_reid_useless, selective_weight_useful_3 = self.style_reid_laye3(x_style_3)
        x_3 = x_IN_3 + x_style_3_reid_useful
        x_3_useless = x_IN_3 + x_style_3_reid_useless

        x_4 = self.layer4(x_3)  # torch.Size([64, 2048, 16, 8])
        x_4_ori = x_4
        x_IN_4 = self.IN4(x_4) #\tilde{F}
        x_style_4 = x_4 - x_IN_4
        x_style_4_reid_useful, x_style_4_reid_useless, selective_weight_useful_4 = self.style_reid_laye4(x_style_4)
        x_4_useful = x_IN_4 + x_style_4_reid_useful # R+
        x_4_useless = x_IN_4 + x_style_4_reid_useless # R-

        return  x_IN_1, x_1, x_1_useless,\
                x_IN_2, x_2, x_2_useless,\
                x_IN_3, x_3, x_3_useless,\
                x_IN_4, x_4_useful, x_4_useless
                    
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()