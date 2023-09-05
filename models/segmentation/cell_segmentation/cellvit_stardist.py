# -*- coding: utf-8 -*-
# class StarDistUnet(BaseMultiTaskSegModel):
#     def __init__(
#         self,
#         type_classes,
#     ) -> None:
#         super().__init__()

#         self.decoders = ("stardist",)
#         self.nrays = 32
#         self.heads = {
#             "stardist": {"stardist": self.n_rays, "dist": 1},
#             "type": {"type": type_classes},
#         }
#         self.inst_key = ("dist",)
#         self.depth = (4,)
#         self.out_channels = ((256, 128, 64, 32),)
#         self.style_channels = (None,)
#         # self.enc_name = "resnet50",
#         # self.enc_pretrain = True,
#         self.enc_freeze = (False,)
#         self.upsampling = ("fixed-unpool",)
#         self.long_skip = ("unet",)
#         self.merge_policy = ("cat",)
#         self.short_skip = ("basic",)
#         self.block_type = ("basic",)
#         self.normalization = (None,)
#         self.activation = ("relu",)
#         self.convolution = ("conv",)
#         self.preactivate = (False,)
#         self.attention = (None,)
#         self.preattend = (False,)
#         self.add_stem_skip = (False,)
#         self.skip_params = (None,)
#         self.encoder_params = (None,)
#         self.input_channels = 3

#         n_layers = (1,) * depth
#         n_blocks = ((2,),) * depth
#         dec_params = {
#             d: _create_stardist_args(
#                 depth,
#                 normalization,
#                 activation,
#                 convolution,
#                 attention,
#                 preactivate,
#                 preattend,
#                 short_skip,
#                 use_style,
#                 block_type,
#                 merge_policy,
#                 skip_params,
#                 upsampling,
#             )
#             for d in decoders
#         }

#         # self.num_tissue_classes = num_tissue_classes
#         # self.num_nuclei_classes = num_nuclei_classes
#         self.encoder = ViTCellViT(
#             patch_size=16,
#             num_classes=self.num_tissue_classes,
#             embed_dim=384,
#             depth=12,
#             num_heads=6,
#             mlp_ratio=4,
#             qkv_bias=True,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             extract_layers=[3, 6, 9, 12],
#             drop_rate=0,
#             attn_drop_rate=0,
#             drop_path_rate=0,
#         )

#         # get the reduction factors for the encoder
#         enc_reductions = tuple([inf["reduction"] for inf in self.encoder.feature_info])

#         # style
#         self.make_style = None
#         if use_style:
#             self.make_style = StyleReshape(self.encoder.out_channels[0], style_channels)

#         # set decoder
#         for decoder_name in decoders:
#             decoder = Decoder(
#                 enc_channels=self.encoder.out_channels,
#                 enc_reductions=enc_reductions,
#                 out_channels=out_channels,
#                 style_channels=style_channels,
#                 long_skip=long_skip,
#                 n_conv_layers=n_layers,
#                 n_conv_blocks=n_blocks,
#                 stage_params=dec_params[decoder_name],
#             )
#             self.add_module(f"{decoder_name}_decoder", decoder)

#         # optional stem skip
#         if add_stem_skip:
#             for decoder_name in decoders:
#                 stem_skip = StemSkip(
#                     out_channels=out_channels[-1],
#                     merge_policy=merge_policy,
#                     n_blocks=2,
#                     short_skip=short_skip,
#                     block_type=block_type,
#                     normalization=normalization,
#                     activation=activation,
#                     convolution=convolution,
#                     attention=attention,
#                     preactivate=preactivate,
#                     preattend=preattend,
#                 )
#                 self.add_module(f"{decoder_name}_stem_skip", stem_skip)

#         # set additional conv blocks ('avoid “fight over features”'.)
#         for decoder_name in extra_convs.keys():
#             for extra_conv, n_channels in extra_convs[decoder_name].items():
#                 features = nn.Conv2d(
#                     in_channels=decoder.out_channels,
#                     out_channels=n_channels,
#                     kernel_size=3,
#                     padding=1,
#                     bias=False,
#                 )
#                 self.add_module(f"{extra_conv}_features", features)

#         # set heads
#         for decoder_name in extra_convs.keys():
#             for extra_conv, in_channels in extra_convs[decoder_name].items():
#                 for output_name, n_classes in heads[extra_conv].items():
#                     seg_head = SegHead(
#                         in_channels=in_channels,
#                         out_channels=n_classes,
#                         kernel_size=1,
#                     )
#                     self.add_module(f"{output_name}_seg_head", seg_head)

#         self.name = f"StardistUnet-{enc_name}"

#         # init decoder weights
#         self.initialize()

#         # freeze encoder if specified
#         if enc_freeze:
#             self.freeze_encoder()

#     def forward(
#         self,
#         x: torch.Tensor,
#         return_feats: bool = False,
#     ) -> Union[
#         Dict[str, torch.Tensor],
#         Tuple[
#             List[torch.Tensor],
#             Dict[str, torch.Tensor],
#             Dict[str, torch.Tensor],
#         ],
#     ]:
#         """Forward pass of Stardist.

#         Parameters
#         ----------
#             x : torch.Tensor
#                 Input image batch. Shape: (B, C, H, W).
#             return_feats : bool, default=False
#                 If True, encoder, decoder, and head outputs will all be returned

#         Returns
#         -------
#         Union[
#             Dict[str, torch.Tensor],
#             Tuple[
#                 List[torch.Tensor],
#                 Dict[str, torch.Tensor],
#                 Dict[str, torch.Tensor],
#             ],
#         ]:
#             Dictionary mapping of output names to outputs or if `return_feats == True`
#             returns also the encoder features in a list, decoder features as a dict
#             mapping decoder names to outputs and the final head outputs dict.
#         """
#         feats, dec_feats = self.forward_features(x)

#         if return_feats:
#             ret_dec_feats = dec_feats.copy()

#         # Extra convs after decoders
#         for e in self.extra_convs.keys():
#             for extra_conv in self.extra_convs[e].keys():
#                 k = self.aux_key if extra_conv not in dec_feats.keys() else extra_conv

#                 dec_feats[extra_conv] = [
#                     self[f"{extra_conv}_features"](dec_feats[k][-1])
#                 ]  # use last decoder feat

#         # seg heads
#         for decoder_name in self.heads.keys():
#             for head_name in self.heads[decoder_name].keys():
#                 k = self.aux_key if head_name not in dec_feats.keys() else head_name
#                 if k != head_name:
#                     dec_feats[head_name] = dec_feats[k]

#         out = self.forward_heads(dec_feats)

#         if return_feats:
#             return feats, ret_dec_feats, out

#         return out
