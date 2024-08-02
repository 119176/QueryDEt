import torch
import math
import numpy as np
 
 
def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    reduction: str = "none",
    xywh: bool = False,
    giou: bool = False,
    diou: bool = False,
    ciou: bool = False,
    eiou: bool = True,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    实现各种IoU
    Parameters
    ----------
    box1        shape(b, c, h, w,4)
    box2        shape(b, c, h, w,4)
    xywh        是否使用中心点和wh，如果是False，输入就是左上右下四个坐标
    GIoU        是否GIoU
    DIoU        是否DIoU
    CIoU        是否CIoU
    EIoU        是否EIoU
    eps         防止除零的小量
    Returns
    -------
    """
    # 获取边界框的坐标
    if xywh:
        # 将 xywh 转换成 xyxy
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
 
    else:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(dim=-1)
 
    # 区域交集
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
    # 区域并集
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    # 计算iou
    iou = inter / union
 
    if giou or diou or ciou or eiou:
        # 计算最小外接矩形的wh
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
 
        if ciou or diou or eiou:
            # 计算最小外接矩形角线的平方
            c2 = cw ** 2 + ch ** 2 + eps
            # 计算最小外接矩形中点距离的平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if diou:
                # 输出DIoU
                IOU = iou - rho2 / c2
            elif ciou:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                # 输出CIoU
                IOU = iou - (rho2 / c2 + v * alpha)
            elif eiou:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                # 输出EIoU
                IOU = iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
        else:
            c_area = cw * ch + eps  # convex area
            # 输出GIoU
            IOU = iou - (c_area - union) / c_area
    else:
        # 输出IoU
       IOU = iou
     
    loss = 1 - IOU
    
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
 