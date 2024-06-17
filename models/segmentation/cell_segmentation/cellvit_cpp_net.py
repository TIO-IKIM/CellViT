# -*- coding: utf-8 -*-
# Adapted from https://github.com/csccsccsccsc/cpp-net/blob/main/cppnet/models/SamplingFeatures2.py # noqa
# File copied from cellsegmodelspytorch
# TODO: check docstring for all models and also file strings at the beginning


import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_segmentation.utils.post_proc_stardist import StarDistPostProcessor

from .cellvit import CellViT, CellViT256, CellViTSAM
from .utils import Conv2DBlock, Deconv2DBlock, ViTCellViT, ViTCellViTDeit


def feature_sampling(
    feature_map: torch.Tensor,
    coord_map: torch.Tensor,
    nd_sampling: int,
    sampling_mode: str = "nearest",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample features from feature map with boundary-pixel coordinates.

    Parameters
    ----------
    feature_map : torch.Tensor
        Input feature map. Shape: (B, C, H, W)
    coord_map : torch.Tensor
        Boundary-pixel coordinate grid. Shape: (B, nrays, 2, H, W)
    nd_sampling : int
        Number of sampling points in each ray.
    sampling_mode : str, optional
        Sampling mode, by default "nearest".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - sampled_features. Shape: (B, K*C', H, W)
        - sampling coords.
            Shape: (nrays*B, H, W, 2) if nd_sampling > 0
            Shape: (B, nrays*H, W, 2) if nd_sampling <= 0
    """
    b, c, h, w = feature_map.shape
    _, nrays, _, _, _ = coord_map.shape

    sampling_coord = coord_map
    sampling_coord[:, :, 0, :, :] = sampling_coord[:, :, 0, :, :] / (w - 1)
    sampling_coord[:, :, 1, :, :] = sampling_coord[:, :, 1, :, :] / (h - 1)
    sampling_coord = sampling_coord * 2.0 - 1.0

    assert nrays * nd_sampling == c

    if nd_sampling > 0:
        sampling_coord = sampling_coord.permute(1, 0, 3, 4, 2)
        sampling_coord = sampling_coord.flatten(start_dim=0, end_dim=1)  # kb, h, w, 2
        sampling_features = F.grid_sample(
            feature_map.view(b, nrays, nd_sampling, h, w)
            .permute(1, 0, 2, 3, 4)
            .flatten(start_dim=0, end_dim=1),
            sampling_coord,
            mode=sampling_mode,
            align_corners=False,
        )  # kb, c', h, w
        sampling_features = sampling_features.view(nrays, b, nd_sampling, h, w).permute(
            1, 0, 2, 3, 4
        )  # b, k, c', h, w
    else:
        sampling_coord = sampling_coord.permute(0, 1, 3, 4, 2).flatten(
            start_dim=1, end_dim=2
        )  # b, kh, w, 2
        sampling_features = F.grid_sample(
            feature_map, sampling_coord, mode=sampling_mode, align_corners=False
        )
        sampling_features = sampling_features.view(b, c, nrays, h, w).permute(
            0, 2, 1, 3, 4
        )  # b, k, c'/c, h, w

    sampling_features = sampling_features.flatten(
        start_dim=1, end_dim=2
    )  # b, k*c', h, w

    return sampling_features, sampling_coord


class SamplingFeatures(nn.Module):
    def __init__(self, nrays: int, sampling_mode: str = "nearest") -> None:
        """Sample features from feature map with boundary-pixel coordinates.

        Parameters
        ----------
        nrays : int
            Number of rays.
        sampling_mode : str, optional
            Sampling mode, by default 'nearest'.
        """
        super().__init__()
        self.nrays = nrays
        self.angles = (
            torch.arange(nrays).float() / float(nrays) * math.pi * 2.0
        )  # 0 - 2*pi
        self.sin_angles = torch.sin(self.angles).view(1, nrays, 1, 1)
        self.cos_angles = torch.cos(self.angles).view(1, nrays, 1, 1)
        self.sampling_mode = sampling_mode

    def forward(
        self, feature_map: torch.Tensor, dist: torch.Tensor, nd_sampling: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample features and coords.

        Parameters
        ----------
        feature_map : torch.Tensor
            Input feature map. Shape: (B, C, H, W)
        dist : torch.Tensor
            Radial distance map. Shape: (B, nrays, H, W)
        nd_sampling : int
            Number of sampling points in each ray.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - sampled_features. Shape: (B, nrays*C, H, W)
            - sampling coords.
                Shape: (nrays*B, H, W, 2) if nd_sampling > 0
                Shape: (B, nrays*H, W, 2) if nd_sampling <= 0
        """
        B, _, H, W = feature_map.shape

        if (
            self.sin_angles.device != dist.device
            or self.cos_angles.device != dist.device
        ):
            self.sin_angles = self.sin_angles.to(dist.device)
            self.cos_angles = self.cos_angles.to(dist.device)

        # sample radial coordinates (full circle) for the rays
        offset_ih = self.sin_angles * dist
        offset_iw = self.cos_angles * dist
        offsets = torch.stack([offset_iw, offset_ih], dim=2)  # (B, nrays, 2, H, W)

        # create a flow/grid (for F.grid_sample)
        x_ = torch.arange(W).view(1, -1).expand(H, -1)
        y_ = torch.arange(H).view(-1, 1).expand(-1, W)
        grid = torch.stack([x_, y_], dim=0).float()

        # (B, 1, 2, H, W)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1).to(dist.device)

        # create the final offset grid
        offsets = offsets + grid

        sampled_features, sampling_coord = feature_sampling(
            feature_map, offsets, nd_sampling, self.sampling_mode
        )

        return sampled_features, sampling_coord, offsets


class CellViTCPP(CellViT):
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
        erosion_factors: Tuple[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
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
        )

        # all decoders here are without a head and return 32 features
        self.stardist_decoder = self.create_upsampling_branch(32)
        self.dist_decoder = self.create_upsampling_branch(32)
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(32)

        self.stardist_head = nn.Conv2d(
            in_channels=32, out_channels=self.nrays, kernel_size=1, bias=False
        )
        self.dist_head = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1, bias=False
        )
        self.type_head = nn.Conv2d(
            in_channels=32,
            out_channels=self.num_nuclei_classes,
            kernel_size=1,
            bias=False,
        )

        self.classifier_head = (
            nn.Linear(self.prompt_embed_dim, num_tissue_classes)
            if num_tissue_classes > 0
            else nn.Identity()
        )

        # cpp-net specific head
        self.erosion_factors = list(erosion_factors)
        self.conv_0_confidence = nn.Conv2d(
            in_channels=32, out_channels=self.nrays, kernel_size=1, bias=False
        )
        self.conv_1_confidence = nn.Conv2d(
            in_channels=(1 + len(erosion_factors)),
            out_channels=(1 + len(erosion_factors)),
            kernel_size=1,
            bias=True,
        )
        self.sampling_features = SamplingFeatures(nrays=self.nrays)
        self.final_activation_ray = nn.ReLU(inplace=True)

    def cppnet_refine(
        self, stardist_map: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine the stardist map and confidence map.

        Parameters
        ----------
            stardist_map : torch.Tensor
                The stardist map. Shape: (B, nrays, H, W)
            features : torch.Tensor
                The features from the encoder. Shape: (B, C, H, W)

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor]
                - refined stardist map. Shape: (B, nrays, H, W)
                - refined confidence map. Shape: (B, C, H, W)
        """
        # cppnet specific ops
        out_confidence = self.conv_0_confidence(features)  # TODO. check feature shape
        out_ray_for_sampling = stardist_map

        ray_refined = [out_ray_for_sampling]
        confidence_refined = [out_confidence]

        for erosion_factor in self.erosion_factors:
            base_dist = (out_ray_for_sampling - 1.0) * erosion_factor
            ray_sampled, _, _ = self.sampling_features(
                out_ray_for_sampling, base_dist, 1
            )
            conf_sampled, _, _ = self.sampling_features(out_confidence, base_dist, 1)
            ray_refined.append(ray_sampled + base_dist)
            confidence_refined.append(conf_sampled)
        ray_refined = torch.stack(ray_refined, dim=1)
        b, k, c, h, w = ray_refined.shape

        confidence_refined = torch.stack(confidence_refined, dim=1)
        confidence_refined = (
            confidence_refined.permute([0, 2, 1, 3, 4])
            .contiguous()
            .view(b * c, k, h, w)
        )
        confidence_refined = self.conv_1_confidence(confidence_refined)
        confidence_refined = confidence_refined.view(b, c, k, h, w).permute(
            [0, 2, 1, 3, 4]
        )
        confidence_refined = F.softmax(confidence_refined, dim=1)

        ray_refined = (ray_refined * confidence_refined).sum(dim=1)
        ray_refined = self.final_activation_ray(ray_refined)

        return ray_refined, confidence_refined

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"

        classifier_logits, _, z = self.encoder(x)

        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        stardist_features = self._forward_upsample(
            z0, z1, z2, z3, z4, self.stardist_decoder
        )
        dist_map_features = self._forward_upsample(
            z0, z1, z2, z3, z4, self.dist_decoder
        )
        type_map_features = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        stardist_head_out = self.stardist_head(stardist_features)
        dist_map_head_out = self.dist_head(dist_map_features)
        type_map_head_out = self.type_head(type_map_features)

        ray_refined, confidence_refined = self.cppnet_refine(
            stardist_head_out, stardist_features
        )

        out_dict = {
            "stardist_map": stardist_head_out,
            "stardist_map_refined": ray_refined,
            "dist_map": dist_map_head_out,
            "nuclei_type_map": type_map_head_out,
            "tissue_types": classifier_logits,
        }

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


class CellViT256CPP(CellViTCPP, CellViT256):
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


class CellViTSAMCPP(CellViTCPP, CellViTSAM):
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
                #TODO: Docstrings
        """
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape of that is divisble by patch_soze (token_size)"

        out_dict = {}

        classifier_logits, _, z = self.encoder(x)

        z0, z1, z2, z3, z4 = x, *z

        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        stardist_features = self._forward_upsample(
            z0, z1, z2, z3, z4, self.stardist_decoder
        )
        dist_map_features = self._forward_upsample(
            z0, z1, z2, z3, z4, self.dist_decoder
        )
        type_map_features = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        stardist_head_out = self.stardist_head(stardist_features)
        dist_map_head_out = self.dist_head(dist_map_features)
        type_map_head_out = self.type_head(type_map_features)

        ray_refined, confidence_refined = self.cppnet_refine(
            stardist_head_out, stardist_features
        )

        out_dict = {
            "stardist_map": stardist_head_out,
            "stardist_map_refined": ray_refined,
            "dist_map": dist_map_head_out,
            "nuclei_type_map": type_map_head_out,
            "tissue_types": self.classifier_head(classifier_logits),
        }

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict


@dataclass
class DataclassCPPStorage:
    """Storing PanNuke Prediction/GT objects for calculating loss, metrics etc. with CPP-Net networks

    Args:
        dist_map (torch.Tensor): Distance map values, bevore Sigmoid Output. Shape: (batch_size, 1, H, W)
        stardist_map (torch.Tensor): Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
        # TODO: docstring
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
    stardist_map_refined: torch.Tensor
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
