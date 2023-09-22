# -*- coding: utf-8 -*-
from models.segmentation.cell_segmentation.cellvit_cpp_net import CellViTCPP

if __name__ == "__main__":
    model = CellViTCPP(6, 19, 384, 3, 12, 6, [3, 6, 9, 12])
