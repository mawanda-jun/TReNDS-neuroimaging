# From https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/master/3DNet/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from collections import OrderedDict

__all__ = [
    'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros((out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)))
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(_BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(_Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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


class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_class=None,
                 dropout_prob=0.,
                 num_init_features=64,
                 in_channels=53,
                 no_cuda=False):

        self.inplanes = num_init_features
        self.no_cuda = no_cuda
        super(ResNet3D, self).__init__()

        # 3D conv net
        self.conv1 = nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3),
                               bias=False)
        num_features = num_init_features  # copy original num_features (64)
        self.bn1 = nn.BatchNorm3d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_features, layers[0], shortcut_type)
        num_features *= 2  # expand num features (128)
        self.layer2 = self._make_layer(
            block, num_features, layers[1], shortcut_type, stride=2)
        num_features *= 2  # expand num features (256)
        self.layer3 = self._make_layer(
            block, num_features, layers[2], shortcut_type, stride=1, dilation=2)
        num_features *= 2  # expand num features (512)
        self.layer4 = self._make_layer(
            block, num_features, layers[3], shortcut_type, stride=1, dilation=4)

        self.fea_dim = num_features * block.expansion

        if num_class is not None:
            self.regressor = nn.Sequential(OrderedDict([
                ('drop', nn.Dropout(dropout_prob)),
                ('regressor', nn.Linear(self.fea_dim, num_class, bias=True))
            ]))
        else:
            self.regressor = None

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        out = x.view((-1, self.fea_dim))
        if self.regressor is not None:
            out = self.regressor(out)
        return out


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(_BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet3d_10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(_BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet3D(_BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet3D(_BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet3D(_Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(_Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(_Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet3D(_Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
