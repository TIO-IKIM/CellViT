# -*- coding: utf-8 -*-
# PyTorch Implementation of Attention Modules
#
# Implementation based on: https://github.com/mahmoodlab/CLAM
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import Tuple
import torch
import torch.nn as nn


class Attention(nn.Module):
    """Basic Attention module. Compare https://github.com/AMLab-Amsterdam/AttentionDeepMIL

    Args:
        in_features (int, optional): Input shape of attention module. Defaults to 1024.
        attention_features (int, optional): Number of attention features. Defaults to 128.
        num_classes (int, optional): Number of output classes. Defaults to 2.
        dropout (bool, optional):  If True, dropout is used. Defaults to False.
        dropout_rate (float, optional): Dropout rate, just applies if dropout parameter is true.
            Needs to be between 0.0 and 1.0. Defaults to 0.25.
    """

    def __init__(
        self,
        in_features: int = 1024,
        attention_features: int = 128,
        num_classes: int = 2,
        dropout: bool = False,
        dropout_rate: float = 0.25,
    ):
        super(Attention, self).__init__()
        # naming
        self.model_name = "AttentionModule"

        # set parameter dimensions for attention
        self.attention_features = attention_features
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.d_rate = dropout_rate

        if self.dropout:
            assert self.d_rate < 1
            self.attention = nn.Sequential(
                nn.Linear(self.in_features, self.attention_features),
                nn.Tanh(),
                nn.Dropout(self.d_rate),
                nn.Linear(self.attention_features, self.num_classes),
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.in_features, self.attention_features),
                nn.Tanh(),
                nn.Linear(self.attention_features, self.num_classes),
            )

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, calculating attention scores for given input vector

        Args:
            H (torch.Tensor): Bag of instances. Shape: (Number of instances, Feature-dimensions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:

            * Attention-Scores
            * H. Shape: Bag of instances. Shape: (Number of instances, Feature-dimensions)
        """
        A = self.attention(H)
        return A, H


class AttentionGated(nn.Module):
    """Gated Attention module. Compare https://github.com/AMLab-Amsterdam/AttentionDeepMIL

    Args:
        in_features (int, optional): Input shape of attention module. Defaults to 1024.
        attention_features (int, optional): Number of attention features. Defaults to 128.
        num_classes (int, optional): Number of output classes. Defaults to 2.
        dropout (bool, optional):  If True, dropout is used. Defaults to False.
        dropout_rate (float, optional): Dropout rate, just applies if dropout parameter is true.
            needs to be between 0.0 and 1.0. Defaults to 0.25.
    """

    def __init__(
        self,
        in_features: int = 1024,
        attention_features: int = 128,
        num_classes: int = 2,
        dropout: bool = False,
        dropout_rate: float = 0.25,
    ):
        super(AttentionGated, self).__init__()
        # naming
        self.model_name = "AttentionModuleGated"

        # set Parameter dimensions for attention
        self.attention_features = attention_features
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.d_rate = dropout_rate

        if self.dropout:
            assert self.d_rate < 1
            self.attention_V = nn.Sequential(
                nn.Linear(self.in_features, self.attention_features),
                nn.Tanh(),
                nn.Dropout(self.d_rate),
            )
            self.attention_U = nn.Sequential(
                nn.Linear(self.in_features, self.attention_features),
                nn.Sigmoid(),
                nn.Dropout(self.d_rate),
            )
            self.attention_W = nn.Sequential(
                nn.Linear(self.attention_features, self.num_classes)
            )

        else:
            self.attention_V = nn.Sequential(
                nn.Linear(self.in_features, self.attention_features), nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(self.in_features, self.attention_features), nn.Sigmoid()
            )
            self.attention_W = nn.Sequential(
                nn.Linear(self.attention_features, self.num_classes)
            )

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, calculating attention scores for given input vector

        Args:
            H (torch.Tensor): Bag of instances. Shape: (Number of instances, Feature-dimensions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:

            * Attention-Scores. Shape: (Number of instances)
            * H. Shape: Bag of instances. Shape: (Number of instances, Feature-dimensions)
        """
        v = self.attention_V(H)
        u = self.attention_U(H)
        A = self.attention_W(v * u)
        return A, H
