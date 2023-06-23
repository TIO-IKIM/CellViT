# -*- coding: utf-8 -*-
# Graph Data model
#
# For more information, please check out docs/readmes/graphs.md
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from dataclasses import dataclass

import torch


@dataclass
class GraphDataWSI:
    """Dataclass for Graph Data

    Args:
        x (torch.Tensor): Node feature matrix with shape (num_nodes, num_nodes_features)
        positions(torch.Tensor):  Each of the objects defined in x has a physical position in a Cartesian coordinate system,
            be it detected cells or extracted patches. That's why we store the 2D position here, globally for the WSI.
            Shape (num_nodes, 2)
        metadata (dict, optional): Metadata about the object is stored here. Defaults to None
    """

    x: torch.Tensor
    positions: torch.Tensor
    metadata: dict
