# -*- coding: utf-8 -*-
from einops import rearrange
from models.encoders.VIT.SAM.image_encoder import ImageEncoderViT
from models.encoders.VIT.vits_histo import VisionTransformer

import torch
import torch.nn as nn
from typing import Callable, Tuple, Type, List


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ViTCellViT(VisionTransformer):
    def __init__(
        self,
        extract_layers: List[int],
        img_size: List[int] = [224],
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        norm_layer: Callable = nn.LayerNorm,
        **kwargs
    ):
        """Vision Transformer with 1D positional embedding

        Args:
            extract_layers: (List[int]): List of Transformer Blocks whose outputs should be returned in addition to the tokens. First blocks starts with 1, and maximum is N=depth.
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch Token size (one dimension only, cause tokens are squared). Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            num_classes (int, optional): Number of output classes. if num classes = 0, raw tokens are returned (nn.Identity).
                Default to 0.
            embed_dim (int, optional): Embedding dimension. Defaults to 768.
            depth(int, optional): Number of Transformer Blocks. Defaults to 12.
            num_heads (int, optional): Number of attention heads per Transformer Block. Defaults to 12.
            mlp_ratio (float, optional): MLP ratio for hidden MLP dimension (Bottleneck = dim*mlp_ratio).
                Defaults to 4.0.
            qkv_bias (bool, optional): If bias should be used for query (q), key (k), and value (v). Defaults to False.
            qk_scale (float, optional): Scaling parameter. Defaults to None.
            drop_rate (float, optional): Dropout in MLP. Defaults to 0.0.
            attn_drop_rate (float, optional): Dropout for attention layer. Defaults to 0.0.
            drop_path_rate (float, optional): Dropout for skip connection. Defaults to 0.0.
            norm_layer (Callable, optional): Normalization layer. Defaults to nn.LayerNorm.

        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.extract_layers = extract_layers

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with returning intermediate outputs for skip connections

        Args:
            x (torch.Tensor): Input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                torch.Tensor: Output of last layers (all tokens, without classification)
                torch.Tensor: Classification output
                torch.Tensor: Skip connection outputs from extract_layer selection
        """
        extracted_layers = []
        x = self.prepare_tokens(x)

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)

        x = self.norm(x)
        output = self.head(x[:, 0])

        return output, x[:, 0], extracted_layers


class ViTCellViTDeit(ImageEncoderViT):
    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            out_chans,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            global_attn_indexes,
        )
        self.extract_layers = extract_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            token_size = x.shape[1]
            x = x + self.pos_embed[:, :token_size, :token_size, :]

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)
        output = self.neck(x.permute(0, 3, 1, 2))
        _output = rearrange(output, "b c h w -> b c (h w)")

        return torch.mean(_output, axis=-1), output, extracted_layers
