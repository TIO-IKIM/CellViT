# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding.

    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        # print(x.shape, padding)
        x = F.pad(x, padding, "constant", 0)
        # print(x.shape)
        return x
