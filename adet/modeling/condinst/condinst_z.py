# -*- coding: utf-8 -*-
import logging

import torch

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.seperators import build_seperator
from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from adet.modeling.condinst import CondInst

__all__ = ["CondInst_Z"]


logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class CondInst_Z(CondInst):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.backbone = build_backbone(cfg, ShapeSpec(channels = cfg.INPUT.STACK_SIZE * len(cfg.MODEL.PIXEL_MEAN)))
        self.seperator = build_seperator(cfg, self.backbone.output_shape())
        self._stack_size = cfg.INPUT.STACK_SIZE

    def forward(self, batched_inputs):
        nb_stacks = len(batched_inputs)
        original_stacks = [None] * nb_stacks
        for s in range(nb_stacks):
            original_stacks[s] = [x["image"].to(self.device) for x in batched_inputs[s]]

        # normalize images
        stacks_norm = [None] * nb_stacks
        for s in range(nb_stacks):
            stacks_norm[s] = torch.stack([self.normalizer(x) for x in original_stacks[s]])
        stacks_norm = ImageList.from_tensors(stacks_norm, self.backbone.size_divisibility)

        tensor_size = stacks_norm.tensor.shape
        concat_stacks = stacks_norm.tensor.view(tensor_size[0], -1, tensor_size[-2], tensor_size[-1])
        features = self.backbone(concat_stacks)

        if "instances" in batched_inputs[0][0]:
            stack_gt_instances = [None] * nb_stacks
            z_gt_instances = [[None] * nb_stacks] * self._stack_size

            for s in range(nb_stacks):
                stack_gt_instances[s] = [x["instances"].to(self.device) for x in batched_inputs[s]]
                if self.boxinst_enabled:
                    original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_stacks[s]]

                    # mask out the bottom area where the COCO dataset probably has wrong annotations
                    for i in range(len(original_image_masks)):
                        im_h = batched_inputs[s][i]["height"]
                        pixels_removed = int(
                            self.bottom_pixels_removed *
                            float(original_stacks[s][i].size(1)) / float(im_h)
                        )
                        if pixels_removed > 0:
                            original_image_masks[i][-pixels_removed:, :] = 0

                    original_stacks[s] = ImageList.from_tensors(original_stacks[s], self.backbone.size_divisibility)
                    original_image_masks = ImageList.from_tensors(
                        original_image_masks, self.backbone.size_divisibility, pad_value=0.0
                    )
                    self.add_bitmasks_from_boxes(
                        stack_gt_instances[s], original_stacks[s].tensor, original_image_masks.tensor,
                        original_stacks[s].tensor.size(-2), original_stacks[s].tensor.size(-1)
                    )
                else:
                    self.add_bitmasks(stack_gt_instances[s], stacks_norm.tensor.size(-2), stacks_norm.tensor.size(-1))

                for z in range(self._stack_size):
                    z_gt_instances[z][s] = stack_gt_instances[s][z]
        else:
            stack_gt_instances = [None] * nb_stacks
            z_gt_instances = [None] * self._stack_size
                
        
        z_features = self.seperator(features)

        if self.training:
            losses = {}

            for z in range(self._stack_size):

                mask_feats, sem_losses = self.mask_branch(z_features[z], z_gt_instances[z])

                proposals, proposal_losses = self.proposal_generator(
                    concat_stacks, z_features[z], z_gt_instances[z], self.controller
                )

                mask_losses = self._forward_mask_heads_train(proposals, mask_feats, z_gt_instances[z])
                
                for l in sem_losses.copy():
                    sem_losses["{}_{}".format(z, l)] = sem_losses.pop(l)
                for l in proposal_losses.copy():
                    proposal_losses["{}_{}".format(z, l)] = proposal_losses.pop(l)
                for l in mask_losses.copy():
                    mask_losses["{}_{}".format(z, l)] = mask_losses.pop(l)
                
                losses.update(sem_losses)
                losses.update(proposal_losses)
                losses.update(mask_losses)

            return losses
                
        else:
            for z in range(self._stack_size):

                mask_feats, sem_losses = self.mask_branch(z_features[z], z_gt_instances[z])

                proposals, proposal_losses = self.proposal_generator(
                    concat_stacks, z_features[z], z_gt_instances[z], self.controller
                )

                pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)

                padded_im_h, padded_im_w = stacks_norm.tensor.size()[-2:]
                processed_results = []
                for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, stacks_norm.image_sizes)):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])

                    instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                    instances_per_im = self.postprocess(
                        instances_per_im, height, width,
                        padded_im_h, padded_im_w
                    )

                    processed_results.append({
                        "instances": instances_per_im
                    })

            return processed_results
