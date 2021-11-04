import torch
from blocks import base_net, get_blk, blk_forward
from prediction import cls_predictor, bbox_predictor, flatten_pred, concat_preds
from data_read import load_data_bananas
from torch import nn
from d2l import torch as d2l
from loss import calc_loss, cls_eval, bbox_eval
from nms import multibox_detection, nms, offset_inverse
from gen_gtboxes import multibox_target
import torchvision
from torch.nn import functional as F
import cv2
import numpy as np


# 定义锚框与图像的比例，底层大尺度特征图采用小比例锚框，用于检测小特征，反之；
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
[0.88, 0.961]]
# 定义锚框长宽比
ratios = [[1, 2, 0.5]] * 5
# 定义每个像素点的锚框数量
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
batch_size = 8
num_epochs = 20
timer = d2l.Timer()
# 导入验证时需要的图片
X1 = torchvision.io.read_image('banana.png').unsqueeze(0).float()
img1 = X1.squeeze(0).permute(1, 2, 0).long()

# 定义SSD算法类
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句 `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X,
            getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                            getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


# 读取训练数据
train_iter, _ = load_data_bananas(batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)


# 开始训练，每个epoch验证一次
for epoch in range(num_epochs):
    metric = d2l.Accumulator(4)
    # 训练模式
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 获得预测结果
        anchors, cls_preds, bbox_preds = net(X)
        # 获得真实的类别和box，并生成相应的掩膜，用于计算锚框回归损失
        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # 计算类别和锚框偏移量的损失
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        # 计算梯度，并更新
        l.mean().backward()
        trainer.step()
        # 累计精度并输出
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on ' f'{str(device)}')
    net.eval()
    with torch.no_grad():
        anchors, cls_preds, bbox_preds = net(X1.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = multibox_detection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        output = output[0, idx]
        d2l.set_figsize((5, 5))
        fig = d2l.plt.imshow(img1)
        for row in output.cpu():
            score = float(row[1])
            if score < 0.9:
                continue
            h, w = img1.shape[0:2]
            bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
            d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
        d2l.plt.show()

