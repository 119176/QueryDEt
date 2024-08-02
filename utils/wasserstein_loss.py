import torch

def wasserstein_lossx1y1x2y2(pred, target, reduction: str = "none", eps=1e-7, constant=12.8):

    pre_width = pred[:, 2] - pred[:, 0]
    pre_higth = pred[:, 3] - pred[:, 1]
    pred_center_x = pred[:, 0] + 0.5 * pre_width
    pred_center_y = pred[:, 1] + 0.5 * pre_higth

    target_width = target[:, 2] - target[:, 0]
    target_higth = target[:, 3] - target[:, 1]
    target_center_x = target[:, 0] + 0.5 * target_width
    target_center_y = target[:, 1] + 0.5 * target_higth

    center1 = torch.stack((pred_center_x, pred_center_y), dim=1)
    center2 = torch.stack((target_center_x, target_center_y), dim=1)

    # 计算中心距离
    # center_distance = ((center1 - center2)**2).sum(dim=1) + eps
    
    whs = center1[:, :2] - center2[:, :2]
    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps

    w1 = pre_width  + eps
    h1 = pre_higth  + eps
    w2 = target_width + eps
    h2 = target_higth + eps

    # 计算宽度和高度距离
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4 



    wasserstein_2 = center_distance + wh_distance
    wasserstein_2_loss = torch.exp(-torch.sqrt(wasserstein_2) / constant)
    loss = 1- wasserstein_2_loss


    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def wasserstein_lossxywh(pred, target, reduction: str = "none", eps=1e-7, constant=12.8):
    """`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_center, y_center, w, h),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    center1 = pred[:, :2]
    center2 = target[:, :2]

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred[:, 2]  + eps
    h1 = pred[:, 3]  + eps
    w2 = target[:, 2] + eps
    h2 = target[:, 3] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    wasserstein_2_loss = torch.exp(-torch.sqrt(wasserstein_2) / constant)
    loss = 1- wasserstein_2_loss
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    
    return loss




