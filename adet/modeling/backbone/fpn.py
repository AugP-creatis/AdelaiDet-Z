from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.modeling.weight_init import init_module

from detectron2.modeling.backbone import FPN, build_resnet_backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from .resnet_lpf import build_resnet_lpf_backbone
from .resnet_interval import build_resnet_interval_backbone
from .mobilenet import build_mnv2_backbone


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet and FCOS to generate extra layers, P6 and P7 from
    C5 or P5 feature.
    """

    def __init__(self, channel_dims, in_channels, out_channels, in_features="res5", *, inter_slice=True):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_features

        if channel_dims == 2:
            self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
            self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        elif channel_dims == 3:
            if inter_slice:
                self.p6 = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                    nn.Conv3d(out_channels, out_channels, (3, 1, 1), (1, 1, 1), (1, 0, 0))
                )
                self.p7 = nn.Sequential(
                    nn.Conv3d(out_channels, out_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                    nn.Conv3d(out_channels, out_channels, (3, 1, 1), (1, 1, 1), (1, 0, 0))
                )
            else:
                self.p6 = nn.Conv3d(out_channels, out_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1))
                self.p7 = nn.Conv3d(out_channels, out_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1))

        for module in [self.p6, self.p7]:
            init_module(module, weight_init.c2_xavier_fill)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, channel_dims, in_channels, out_channels, in_features="res5", *, inter_slice=True):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features

        if channel_dims == 2:
            self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        elif channel_dims == 3:
            if inter_slice:
                self.p6 = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                    nn.Conv3d(out_channels, out_channels, (3, 1, 1), (1, 1, 1), (1, 0, 0))
                )
            else:
                self.p6 = nn.Conv3d(in_channels, out_channels, (1, 3, 3), (1, 2, 2), (0, 1, 1))
            
        for module in [self.p6]:
            init_module(module, weight_init.c2_xavier_fill)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


@BACKBONE_REGISTRY.register()
def build_fcos_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    channel_dims = cfg.MODEL.BACKBONE.DIM

    if channel_dims == 2:
        if cfg.MODEL.BACKBONE.ANTI_ALIAS:
            bottom_up = build_resnet_lpf_backbone(cfg, input_shape)
        elif cfg.MODEL.RESNETS.DEFORM_INTERVAL > 1:
            bottom_up = build_resnet_interval_backbone(cfg, input_shape)
        elif cfg.MODEL.MOBILENET:
            bottom_up = build_mnv2_backbone(cfg, input_shape)
        else:
            bottom_up = build_resnet_backbone(cfg, input_shape)

    elif channel_dims == 3:
        bottom_up = build_resnet_backbone(cfg, input_shape)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels
    inter_slice = cfg.MODEL.BACKBONE.INTER_SLICE

    if top_levels == 2:
        top_block = LastLevelP6P7(channel_dims, in_channels_top, out_channels, "p5", inter_slice=inter_slice)
    if top_levels == 1:
        top_block = LastLevelP6(channel_dims, in_channels_top, out_channels, "p5", inter_slice=inter_slice)
    elif top_levels == 0:
        top_block = None

    backbone = FPN(
        channel_dims=channel_dims,
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
