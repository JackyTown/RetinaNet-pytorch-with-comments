from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    # Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。
    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        # x:input
        # t:target
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21] 20类objects + 背景
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        # p,t 均为list, 相乘得到每一类的分数
        # 如果t > 0, 表明该类别是实际类别, p越大, pt越大
        # 如果t == 0, 表明该类别不是实际类别, p越小, 1-p越大, pt越大
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        # alpha设置为0.25
        # 正样本: w = alpha
        # 负样本: w = 1- alpha
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        # 对于正类样本而言，预测结果为0.95肯定是简单样本，所以(1-0.95)的gamma次方就会很小，这时损失函数值就变得更小，而预测概率为0.3的样本其损失相对很大。对于负类样本而言同样，预测0.1的结果应当远比预测0.4的样本损失值要小得多(0.1的gamma次方会很小)。所以更加关注于难以区分的样本。
        w = w * (1-pt).pow(gamma)
        # 综上，最关注的是0.3-0.4的背景框(难样挖掘)。
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors] 用来预测object的anchor被置为1
        num_pos = pos.data.long().sum() 
        # sum(), 对所有元素的求和(统计负责预测object的anchor总数量)

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        # exclude 背景
        # 在loc_preds中,只有mask对应位置为1的元素,才会被提取出来
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds) # [N,#anchors,21]
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes) 
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # loc_loss and cls_loss
        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (loc_loss+cls_loss)/num_pos # 总损失
        return loss
