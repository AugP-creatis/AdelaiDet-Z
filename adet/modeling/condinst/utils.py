import torch

def mask_nms(seg_masks, scores, iou_threshold):    #inspired by solov2 matrix_nms
    seg_masks = seg_masks.clone().detach()
    sum_masks = seg_masks.sum((1, 2)).float()

    n_samples = len(seg_masks)
    if n_samples == 0:
        return []
    
    seg_masks = seg_masks.reshape(n_samples, -1).float()

    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_y = sum_masks[:, None].expand(n_samples, n_samples)
    union_matrix = sum_masks_y + sum_masks_y.transpose(1, 0) - inter_matrix
    # iou.
    iou_matrix = inter_matrix / union_matrix

    scores = scores[:, None].expand(n_samples, n_samples)

    keep = torch.all(torch.logical_or(iou_matrix.fill_diagonal_(0) < iou_threshold, scores >= scores.transpose(1, 0)), dim=1)

    return keep
