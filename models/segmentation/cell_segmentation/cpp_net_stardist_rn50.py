# -*- coding: utf-8 -*-
# Code adapted from CPP-Net paper and repo
# CPP-net paper and code: https://github.com/csccsccsccsc/cpp-net
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from cell_segmentation.utils.post_proc_stardist import StarDistPostProcessor


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        num_groups = out_ch // 8
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_channels=out_ch, num_groups=num_groups),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up2(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up2, self).__init__()

        if bilinear:
            self.conv_trans = nn.Conv2d(in_ch, in_ch // 2, 1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(self.conv_trans(x1))
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def feature_sampling(feature_map, coord_map, nd_sampling, sampling_mode="nearest"):
    b, c, h, w = feature_map.shape
    # coord_map: b, k, 2, h, w
    # 'k' for k rays in each image
    _, k, _, h, w = coord_map.shape
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    # grid: b, 1, 2, h, w
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1).cuda()
    # sampling_coord: b, k, 2, h, w
    sampling_coord = grid + coord_map
    sampling_coord[:, :, 0, :, :] = sampling_coord[:, :, 0, :, :] / (w - 1)
    sampling_coord[:, :, 1, :, :] = sampling_coord[:, :, 1, :, :] / (h - 1)
    sampling_coord = sampling_coord * 2.0 - 1.0

    assert k * nd_sampling == c

    if nd_sampling > 0:
        sampling_coord = sampling_coord.permute(1, 0, 3, 4, 2).flatten(
            start_dim=0, end_dim=1
        )  # kb, h, w, 2
        sampling_features = F.grid_sample(
            feature_map.view(b, k, nd_sampling, h, w)
            .permute(1, 0, 2, 3, 4)
            .flatten(start_dim=0, end_dim=1),
            sampling_coord,
            mode=sampling_mode,
        )  # kb, c', h, w
        sampling_features = sampling_features.view(k, b, nd_sampling, h, w).permute(
            1, 0, 2, 3, 4
        )  # b, k, c', h, w
    else:
        sampling_coord = sampling_coord.permute(0, 1, 3, 4, 2).flatten(
            start_dim=1, end_dim=2
        )  # b, kh, w, 2
        sampling_features = F.grid_sample(
            feature_map, sampling_coord, mode=sampling_mode
        )
        sampling_features = sampling_features.view(b, c, k, h, w).permute(
            0, 2, 1, 3, 4
        )  # b, k, c'/c, h, w

    sampling_features = sampling_features.flatten(
        start_dim=1, end_dim=2
    )  # b, k*c', h, w

    return sampling_features, sampling_coord


class SamplingFeatures(nn.Module):
    def __init__(self, n_rays, sampling_mode="nearest"):
        super(SamplingFeatures, self).__init__()
        self.n_rays = n_rays
        self.angles = (
            torch.arange(n_rays).float() / float(n_rays) * math.pi * 2.0
        )  # 0 - 2*pi
        self.sin_angles = torch.sin(self.angles).cuda().view(1, n_rays, 1, 1)
        self.cos_angles = torch.cos(self.angles).cuda().view(1, n_rays, 1, 1)
        self.sampling_mode = sampling_mode

    def forward(self, feature_map, dist, nd_sampling):
        # feature_map: b, c, h, w
        # dist: b, k, h, w
        # sampled_features: b, k*c, h, w
        offset_ih = self.sin_angles * dist
        offset_iw = self.cos_angles * dist
        offsets = torch.stack([offset_iw, offset_ih], dim=2)
        sampled_features, sampling_coord = feature_sampling(
            feature_map, offsets, nd_sampling, self.sampling_mode
        )
        return sampled_features, sampling_coord


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return [x1, x2, x3, x4]

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict_for_load = {}
        target_state_dict_keys = model.state_dict().keys()
        for key, value in state_dict.items():
            if key in target_state_dict_keys:
                state_dict_for_load.update({key: value})
                # print('Key ' + key + ' loaded')
        model.load_state_dict(state_dict_for_load)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


class StarDistRN50(nn.Module):
    def __init__(self, n_rays=32, n_seg_cls=6):
        super(StarDistRN50, self).__init__()

        self.encoder = resnet50(True)
        self.up1 = up(2048 + 1024, 1024, bilinear=True)
        self.up2 = up(1024 + 512, 512, bilinear=True)
        self.up3 = up(512 + 256, 256, bilinear=True)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.features = nn.Conv2d(256, 256, 3, padding=1)
        self.out_prob = outconv(256, 1)
        self.out_ray = outconv(256, n_rays)

        self.n_seg_cls = n_seg_cls
        self.up1_seg = up(2048 + 1024, 1024, bilinear=True)
        self.up2_seg = up(1024 + 512, 512, bilinear=True)
        self.up3_seg = up(512 + 256, 256, bilinear=True)
        self.up4_seg = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.out_seg = outconv(256, n_seg_cls)

        self.final_activation_ray = nn.ReLU()
        self.n_rays = n_rays

    def forward(self, img, gt_dist=None):
        x1, x2, x3, x4 = self.encoder(img)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x)
        x = self.features(x)

        out_prob = self.out_prob(x)

        out_ray = self.out_ray(x)
        out_ray = self.final_activation_ray(out_ray)

        x_seg = self.up1_seg(x4, x3)
        x_seg = self.up2_seg(x_seg, x2)
        x_seg = self.up3_seg(x_seg, x1)
        x_seg = self.up4_seg(x_seg)
        out_seg = self.out_seg(x_seg)

        out_dict = {}
        out_dict["dist_map"] = out_prob
        out_dict["stardist_map"] = out_ray
        out_dict["nuclei_type_map"] = out_seg

        return out_dict

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if hasattr(m, "bias"):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.conv_1_confidence.conv.bias, 1.0)

    def freeze_encoder(self):
        """Freeze encoder to not train it"""
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":  # do not freeze head
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model"""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def calculate_instance_map(
        self,
        dist_map: torch.Tensor,
        stardist_map: torch.Tensor,
        nuclei_type_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[dict], torch.Tensor]:
        """Calculate binary nuclei prediction map, nuclei dict and nuclei type map

        Args:
            dist_map (torch.Tensor): Distance probabilities. Shape: (B, 1, H, W)
            stardist_map (torch.Tensor): Stardist probabilities. Shape: (B, n_rays, H, W)
            nuclei_type_map (torch.Tensor): Nuclei type map. Shape: (B, num_nuclei_types, H, W)

        Returns:
            Tuple[torch.Tensor, List[dict], torch.Tensor]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
                * nuclei-instance predictions with shape (B, num_nuclei_types, H, W)
        """
        b, n, h, w = nuclei_type_map.shape
        cell_post_processor = StarDistPostProcessor(nr_types=n, image_shape=(h, w))
        instance_preds = []
        type_preds = []
        instance_type_preds = []

        for i in range(dist_map.shape[0]):
            (
                instance_pred,
                type_pred,
                instance_type_pred,
            ) = cell_post_processor.post_proc_stardist(
                dist_map[i].squeeze().detach().cpu().numpy().astype(np.float32),
                stardist_map[i].detach().cpu().numpy().astype(np.float32),
                nuclei_type_map[i].detach().cpu().numpy().astype(np.float32),
            )
            instance_preds.append(instance_pred)
            type_preds.append(type_pred)
            instance_type_preds.append(instance_type_pred)

        return torch.stack(instance_preds), type_preds, torch.stack(instance_type_preds)
