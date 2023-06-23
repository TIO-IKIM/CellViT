# -*- coding: utf-8 -*-
# Helper functions for models
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from torch import nn


def reset_weights(model: nn.Module) -> None:
    """Reset the parameters of the model to avaid weight leakage

    Args:
        model (nn.Module): PyTorch Model
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def initialize_weights(module: nn.Module) -> None:
    """Initialize Module weights according to xavier

    Args:
        module (nn.Module): Model
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
