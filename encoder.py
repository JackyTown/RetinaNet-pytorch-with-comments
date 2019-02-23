'''Encode object boxes and labels.'''
import math
import torch

from utils import meshgrid, box_iou, box_nms, change_box_order


class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7  
        # 这组anchor_area，是还原到原图尺寸的。分辨率高的feature map(p3)，感受野小，因此对应的anchor_area就小。
        self.aspect_ratios = [1/2., 1/1., 2/1.] # 宽高比
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)] # anchor放缩比，每一层多几种size的anchor。
        self.anchor_wh = self._get_anchor_wh() # 包含所有anchor width and height的tensor

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map. 

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas) # feature map的层数
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2) # [num_fms, anchors in this fm, w and h]

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes 参考fpn每层的分辨率
        # x.ceil()：返回数字的上入整数
        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i] 
            grid_size = input_size / fm_size # feature map中的特征点坐标xy, 还原到原图的比例
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1]) # 得到feature map的长宽  
            xy = meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            # meshgrid(x,y) 得到所有特征点的坐标
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2) 
            # x.expand()：扩展某个size为1的维度 扩展方式：自拷贝 
            # 9个不同形状的anchor，xy是相同的
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            # 每个特征点对应的anchor_size是相同的
            box = torch.cat([xy,wh], 3)  # [fm_h,fm_w,9,2] -> [fm_h,fm_w,9,4] (x,y) + (w,h) -> (x,y,w,h)
            # cat(): 指定维度上的堆叠
            boxes.append(box.view(-1,4)) # [fm_h * fm_w * 9, 4]
        return torch.cat(boxes, 0)  # [num_fms * fm_h * fm_w * 9, 4]

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size) # [w, h]
        anchor_boxes = self._get_anchor_boxes(input_size) # [num_fms * fm_h * fm_w * 9, 4]
        boxes = change_box_order(boxes, 'xyxy2xywh') # (xmin,ymin,xmax,ymax) -> (xcenter,ycenter,width,height)

        ious = box_iou(anchor_boxes, boxes, order='xywh') # [N,M]
        max_ious, max_ids = ious.max(1) # 针对每行(anchor)，找出最大值max_iou及相应的列索引max_id
        # max_ious [N,]: N个anchor对应的最大IOU
        # max_ids: 和N个anchor拥有最大IOU的box的id
        boxes = boxes[max_ids] # [N,4] 和N个anchor拥有最大IOU的box的(xcenter,ycenter,width,height)
        '''
          boxes = [tensor([[1., 1., 3., 3.],
          [0., 0., 2., 2.]])]
          max_ids = tensor([1, 0, 0])
          boxes[max_ids] = tensor([[0., 0., 2., 2.],
          [1., 1., 3., 3.],
          [1., 1., 3., 3.]])
        '''

        # encoding
        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        '''
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
        '''
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        '''
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
        '''
        loc_targets = torch.cat([loc_xy,loc_wh], 1) # [N,4]
        cls_targets = 1 + labels[max_ids] # 所有类别值加一，背景的类别值为0

        cls_targets[max_ious<0.5] = 0 # 这些anchor不负责预测object，其receptive field对应的是背景
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = 0.5
        NMS_THRESH = 0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2] # 得到预测的xy
        loc_wh = loc_preds[:,2:] # 得到预测的wh

        # decoding (encoding的逆操作)
        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4] 
        # (xmin,ymin,xmax,ymax) -> (xcenter,ycenter,width,height)

        score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        ids = score > CLS_THRESH # 若元素大于CLS_THRESH, 置为1, 反之为0 
        ids = ids.nonzero().squeeze()# 得到非零元素的索引，并压为一维 -> 保留分数大于阈值的prediction_box的id # [#obj,]
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH) # NMS
        return boxes[ids][keep], labels[ids][keep]
