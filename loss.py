import torch
from torch import nn


cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

# 定义损失函数，类别误差采用交叉熵损失函数，锚框偏差采用L1损失函数
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # print(cls_preds.reshape(-1).reshape(batch_size, num_classes, -1).size())
    # print(cls_labels.reshape(-1).reshape(batch_size, -1).size())
    # cls = cls_loss(cls_preds.reshape(-1, num_classes),
    #                cls_labels.reshape(-1).reshape(batch_size, -1)).mean(dim=1)
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())