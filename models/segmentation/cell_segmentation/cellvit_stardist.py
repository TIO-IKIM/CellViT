# -*- coding: utf-8 -*-
# CellViT networks and adaptions based on StarDist, without sharing encoders
#
# UNETR paper and code: https://github.com/tamasino52/UNETR
# SAM paper and code: https://segment-anything.com/
# StarDist paper and code: https://github.com/stardist/stardist
# CPP-net paper and code: https://github.com/csccsccsccsc/cpp-net
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from cell_segmentation.utils.post_proc_stardist import StarDistPostProcessor

from .cellvit import CellViT, CellViT256, CellViTSAM
from .utils import Conv2DBlock, Deconv2DBlock, ViTCellViT, ViTCellViTDeit


class CellViTStarDist(CellViT):
    """CellViT Modell with StarDist heads (separated decoder).

    Skip connections are shared between branches, but each network has a distinct encoder

    The moodell is having four branches:
        * tissue_types: Tissue prediction based on global class token
        * nuclei_type_map: Nuclei instance-prediction
        * stardist_map: Stardist mapping
        * dist_map: Probability distance mapping

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        embed_dim (int): Embedding dimension of backbone ViT
        input_channels (int): Number of input channels
        depth (int): Depth of the backbone ViT
        num_heads (int): Number of heads of the backbone ViT
        extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens. First blocks starts with 1, and maximum is N=depth.
            Is used for skip connections. At least 4 skip connections needs to be returned.
        nrays (int, optional): Number of stardist nray vectors. Default to 32.
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone ViT. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone ViT. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
    """

    def __init__(
        self,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        embed_dim: int,
        input_channels: int,
        depth: int,
        num_heads: int,
        extract_layers: List,
        nrays: int = 32,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        super(CellViT, self).__init__()
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.patch_size = 16
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.nrays = nrays
        self.prompt_embed_dim = 256

        self.encoder = ViTCellViT(
            patch_size=self.patch_size,
            num_classes=self.num_tissue_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            extract_layers=self.extract_layers,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        self.branches_output = {
            "stardist_map": self.nrays,
            "dist_map": 1,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.stardist_decoder = self.create_upsampling_branch(
            self.branches_output["stardist_map"]
        )
        self.stardist_activation_function = nn.ReLU()  # TODO: rename

        self.dist_decoder = self.create_upsampling_branch(
            self.branches_output["dist_map"]
        )
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes
        )
        self.classifier_head = (
            nn.Linear(self.prompt_embed_dim, num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * stardist_map: Stardist map. Shape (B, n_rays, H, W)
                * dist_map: Distance probabilities. Shape: (B, 1, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
        """
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisible by patch_size (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = classifier_logits

        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        out_dict["stardist_map"] = self.stardist_activation_function(
            self._forward_upsample(z0, z1, z2, z3, z4, self.stardist_decoder)
        )  # TODO: rename
        out_dict["dist_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.dist_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )
        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

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


class CellViT256StarDist(CellViTStarDist, CellViT256):
    """CellViT Modell with StarDist heads (separated decoder)
    with ViT-256 backbone settings (https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth)

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        model256_path (Union[Path, str]): Path to ViT 256 backbone model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        nrays (int, optional): Number of stardist nray vectors
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
    """

    def __init__(
        self,
        model256_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        nrays: int = 32,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        self.patch_size = 16
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.extract_layers = [3, 6, 9, 12]
        self.input_channels = 3  # RGB
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.nrays = nrays

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            nrays=self.nrays,
        )

        self.model256_path = model256_path


class CellViTSAMStarDist(CellViTStarDist, CellViTSAM):
    """CellViT Modell with StarDist heads (separated decoder) with SAM backbone settings

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        model_path (Union[Path, str]): Path to pretrained SAM model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM model type
        nrays (int, optional): _description_. Defaults to 32.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.

    Raises:
        NotImplementedError: Unknown SAM configuration
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["SAM-B", "SAM-L", "SAM-H"],
        nrays: int = 32,
        drop_rate: float = 0,
    ):
        if vit_structure.upper() == "SAM-B":
            self.init_vit_b()
        elif vit_structure.upper() == "SAM-L":
            self.init_vit_l()
        elif vit_structure.upper() == "SAM-H":
            self.init_vit_h()
        else:
            raise NotImplementedError("Unknown ViT-SAM backbone structure")

        self.input_channels = 3  # RGB
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.num_nuclei_classes = num_nuclei_classes
        self.model_path = model_path

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            nrays=nrays,
        )

        self.prompt_embed_dim = 256

        self.encoder = ViTCellViTDeit(
            extract_layers=self.extract_layers,
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
        )

        self.classifier_head = (
            nn.Linear(self.prompt_embed_dim, num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )
        self.final_activation_ray = nn.ReLU()

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * stardist_map: Stardist map. Shape (B, n_rays, H, W)
                * dist_map: Distance probabilities. Shape: (B, 1, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * (optinal) tokens
        """
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        out_dict["stardist_map"] = self.final_activation_ray(
            self._forward_upsample(z0, z1, z2, z3, z4, self.stardist_decoder)
        )
        out_dict["dist_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.dist_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict


@dataclass
class DataclassStarDistStorage:
    """Storing PanNuke Prediction/GT objects for calculating loss, metrics etc. with StarDist networks

    Args:
        dist_map (torch.Tensor): Distance map values, bevore Sigmoid Output. Shape: (batch_size, 1, H, W)
        stardist_map (torch.Tensor): Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
        nuclei_type_map (torch.Tensor): Softmax output for nuclei type-prediction. Shape: (batch_size, num_tissue_classes, H, W)
        batch_size (int): Batch size of the experiment
        dist_map_sigmoid (torch.Tensor, optional): Distance map values, after Sigmoid Output. Shape: (batch_size, 1, H, W). Defaults to None.
        instance_map (torch.Tensor, optional): Pixel-wise nuclear instance segmentation.
            Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
            Defaults to None.
        instance_types_nuclei (torch.Tensor, optional): Pixel-wise nuclear instance segmentation predictions, for each nuclei type.
            Each instance has its own integer, starting from 1.
            Shape: (batch_size, num_nuclei_classes, H, W)
            Defaults to None.
        instance_types (list, optional): Instance type prediction list.
            Each list entry stands for one image. Each list entry is a dictionary with the following structure:
            Main Key is the nuclei instance number (int), with a dict as value.
            For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
            contour, type_prob (probability), type (nuclei type)
            Defaults to None.
        tissue_types (torch.Tensor, optional): Logit tissue prediction output. Shape: (batch_size, num_tissue_classes).
            Defaults to None.
        h (int, optional): Height of used input images. Defaults to 256.
        w (int, optional): Width of used input images. Defaults to 256.
        num_tissue_classes (int, optional): Number of tissue classes in the data. Defaults to 19.
        num_nuclei_classes (int, optional): Number of nuclei types in the data (including background). Defaults to 6.
    """

    dist_map: torch.Tensor
    stardist_map: torch.Tensor
    nuclei_type_map: torch.Tensor
    batch_size: int
    dist_map_sigmoid: torch.Tensor = None
    instance_map: torch.Tensor = None
    instance_types_nuclei: torch.Tensor = None
    instance_types: list = None
    tissue_types: torch.Tensor = None
    h: int = 256
    w: int = 256
    num_tissue_classes: int = 19
    num_nuclei_classes: int = 6

    def get_dict(self):
        return self.__dict__
