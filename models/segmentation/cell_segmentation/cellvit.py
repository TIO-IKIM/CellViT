# -*- coding: utf-8 -*-
# CellViT networks and adaptions, without sharing encoders
#
# UNETR paper and code: https://github.com/tamasino52/UNETR
# SAM paper and code: https://segment-anything.com/
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from cell_segmentation.utils.post_proc_cellvit import DetectionCellPostProcessor

from .utils import Conv2DBlock, Deconv2DBlock, ViTCellViT, ViTCellViTDeit


class CellViT(nn.Module):
    """CellViT Modell for cell segmentation. U-Net like network with vision transformer as backbone encoder

    Skip connections are shared between branches, but each network has a distinct encoder

    The modell is having multiple branches:
        * tissue_types: Tissue prediction based on global class token
        * nuclei_binary_map: Binary nuclei prediction
        * hv_map: HV-prediction to separate isolated instances
        * nuclei_type_map: Nuclei instance-prediction
        * [Optional, if regression loss]:
        * regression_map: Regression map for binary prediction

    Args:
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        embed_dim (int): Embedding dimension of backbone ViT
        input_channels (int): Number of input channels
        depth (int): Depth of the backbone ViT
        num_heads (int): Number of heads of the backbone ViT
        extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens. First blocks starts with 1, and maximum is N=depth.
            Is used for skip connections. At least 4 skip connections needs to be returned.
        mlp_ratio (float, optional): MLP ratio for hidden MLP dimension of backbone ViT. Defaults to 4.
        qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v) in backbone ViT. Defaults to True.
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
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
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,
    ):
        # For simplicity, we will assume that extract layers must have a length of 4
        super().__init__()
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

        self.regression_loss = regression_loss
        offset_branches = 0
        if self.regression_loss:
            offset_branches = 2
        self.branches_output = {
            "nuclei_binary_map": 2 + offset_branches,
            "hv_map": 2,
            "nuclei_type_maps": self.num_nuclei_classes,
        }

        self.nuclei_binary_map_decoder = self.create_upsampling_branch(
            2 + offset_branches
        )  # todo: adapt for helper loss
        self.hv_map_decoder = self.create_upsampling_branch(
            2
        )  # todo: adapt for helper loss
        self.nuclei_type_maps_decoder = self.create_upsampling_branch(
            self.num_nuclei_classes
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
                * [Optional, if regression loss]:
                * regression_map: Regression map for binary prediction. Shape: (B, 2, H, W)
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

        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )
        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        """Forward upsample branch

        Args:
            z0 (torch.Tensor): Highest skip
            z1 (torch.Tensor): 1. Skip
            z2 (torch.Tensor): 2. Skip
            z3 (torch.Tensor): 3. Skip
            z4 (torch.Tensor): Bottleneck
            branch_decoder (nn.Sequential): Branch decoder network

        Returns:
            torch.Tensor: Branch Output
        """
        b4 = branch_decoder.bottleneck_upsampler(z4)
        b3 = self.decoder3(z3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        b2 = self.decoder2(z2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        b1 = self.decoder1(z1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        b0 = self.decoder0(z0)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
        self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)

    def freeze_encoder(self):
        """Freeze encoder to not train it"""
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":  # do not freeze head
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder to train the whole model"""
        for p in self.encoder.parameters():
            p.requires_grad = True


class CellViT256(CellViT):
    """CellViT with ViT-256 backbone settings (https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth)

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        model256_path (Union[Path, str]): Path to ViT 256 backbone model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.
    """

    def __init__(
        self,
        model256_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        regression_loss: bool = False,  # to use regressive loss for predicting vector components
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
            regression_loss=regression_loss,
        )

        self.model256_path = model256_path

    def load_pretrained_encoder(self, model256_path: str):
        """Load pretrained ViT-256 from provided path

        Args:
            model256_path (str): Path to ViT-256
        """
        state_dict = torch.load(str(model256_path), map_location="cpu")["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")


class CellViTSAM(CellViT):
    """CellViT with SAM backbone settings

    Skip connections are shared between branches, but each network has a distinct encoder

    Args:
        model_path (Union[Path, str]): Path to pretrained SAM model
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM model type
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary decoder, but returns it as own entry in dict. Defaults to False.

    Raises:
        NotImplementedError: Unknown SAM configuration
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["SAM-B", "SAM-L", "SAM-H"],
        drop_rate: float = 0,
        regression_loss: bool = False,
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
            regression_loss=regression_loss,
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

    def load_pretrained_encoder(self, model_path):
        """Load pretrained SAM encoder from provided path

        Args:
            model_path (str): Path to SAM model
        """
        state_dict = torch.load(str(model_path), map_location="cpu")
        image_encoder = self.encoder
        msg = image_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")
        self.encoder = image_encoder

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        """Forward pass

        Args:
            x (torch.Tensor): Images in BCHW style
            retrieve_tokens (bool, optional): If tokens of ViT should be returned as well. Defaults to False.

        Returns:
            dict: Output for all branches:
                * tissue_types: Raw tissue type prediction. Shape: (B, num_tissue_classes)
                * nuclei_binary_map: Raw binary cell segmentation predictions. Shape: (B, 2, H, W)
                * hv_map: Binary HV Map predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Raw binary nuclei type preditcions. Shape: (B, num_nuclei_classes, H, W)
                * [Optional, if retrieve tokens]: tokens
                * [Optional, if regression loss]:
                * regression_map: Regression map for binary prediction. Shape: (B, 2, H, W)
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

        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )

        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def init_vit_b(self):
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]

    def init_vit_l(self):
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]

    def init_vit_h(self):
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32]


@dataclass
class DataclassHVStorage:
    """Storing PanNuke Prediction/GT objects for calculating loss, metrics etc. with HoverNet networks

    Args:
        nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
        hv_map (torch.Tensor): Logit output for HV-Map. Shape: (batch_size, 2, H, W)
        nuclei_type_map (torch.Tensor): Softmax output for nuclei type-prediction. Shape: (batch_size, num_tissue_classes, H, W)
        tissue_types (torch.Tensor): Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
        instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
            Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
        instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type.
            Each instance has its own integer, starting from 1.
            Shape: (batch_size, num_nuclei_classes, H, W)
        batch_size (int): Batch size of the experiment
        instance_types (list, optional): Instance type prediction list.
            Each list entry stands for one image. Each list entry is a dictionary with the following structure:
            Main Key is the nuclei instance number (int), with a dict as value.
            For each instance, the dictionary contains the keys: bbox (bounding box), centroid (centroid coordinates),
            contour, type_prob (probability), type (nuclei type)
            Defaults to None.
        regression_map (torch.Tensor, optional): Regression map for binary prediction map.
            Shape: (batch_size, 2, H, W). Defaults to None.
        regression_loss (bool, optional): Indicating if regression map is present. Defaults to False.
        h (int, optional): Height of used input images. Defaults to 256.
        w (int, optional): Width of used input images. Defaults to 256.
        num_tissue_classes (int, optional): Number of tissue classes in the data. Defaults to 19.
        num_nuclei_classes (int, optional): Number of nuclei types in the data (including background). Defaults to 6.
    """

    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    tissue_types: torch.Tensor
    nuclei_type_map: torch.Tensor
    instance_map: torch.Tensor
    instance_types_nuclei: torch.Tensor
    batch_size: int
    instance_types: list = None
    regression_map: torch.Tensor = None
    regression_loss: bool = False
    h: int = 256
    w: int = 256
    num_tissue_classes: int = 19
    num_nuclei_classes: int = 6

    # def __post_init__(self):
    #     # check shape of every element
    #     assert list(self.nuclei_binary_map.shape) == [
    #         self.batch_size,
    #         2,
    #         self.h,
    #         self.w,
    #     ], "Nuclei Binary Map must be a softmax tensor with shape (B, 2, H, W)"
    #     assert list(self.hv_map.shape) == [
    #         self.batch_size,
    #         2,
    #         self.h,
    #         self.w,
    #     ], "HV Map must be a tensor with shape (B, 2, H, W)"
    #     assert list(self.nuclei_type_map.shape) == [
    #         self.batch_size,
    #         self.num_nuclei_classes,
    #         self.h,
    #         self.w,
    #     ], "Nuclei Type Map must be a tensor with shape (B, num_nuclei_classes, H, W)"
    #     assert list(self.instance_map.shape) == [
    #         self.batch_size,
    #         self.h,
    #         self.w,
    #     ], "Instance Map must be a tensor with shape (B, H, W)"
    #     assert list(self.instance_types_nuclei.shape) == [
    #         self.batch_size,
    #         self.num_nuclei_classes,
    #         self.h,
    #         self.w,
    #     ], "Instance Types Nuclei must be a tensor with shape (B, num_nuclei_classes, H, W)"
    #     if self.regression_map is not None:
    #         self.regression_loss = True
    #     else:
    #         self.regression_loss = False

    def get_dict(self) -> dict:
        """Return dictionary of entries"""
        property_dict = self.__dict__
        if not self.regression_loss and "regression_map" in property_dict.keys():
            property_dict.pop("regression_map")
        return property_dict
