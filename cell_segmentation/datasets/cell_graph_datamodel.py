# -*- coding: utf-8 -*-
# Graph Data model
#
# For more information, please check out docs/readmes/graphs.md
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from dataclasses import dataclass
from typing import List

import torch

from datamodel.graph_datamodel import GraphDataWSI


@dataclass
class CellGraphDataWSI(GraphDataWSI):
    """Dataclass for Graph Data

    Args:
        contours (List[torch.Tensor]): Contour Data for each object.
    """

    contours: List[torch.Tensor]
