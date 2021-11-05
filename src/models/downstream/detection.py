from collections import OrderedDict
from itertools import chain
from typing import List, Dict, Union

import torch
from torch import nn
from torch.cuda.amp import autocast

from models.image.resnet import ResNetFeatureExtractor
from models.image.resunet import ResUNetFeatureExtractor
from models.image.unet import UNetFeatureExtractor


class YOLOv3WithResNetBackbone(nn.Module):
    """
    https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
    https://github.com/BobLiu20/YOLOv3_PyTorch
    https://github.com/BobLiu20/YOLOv3_PyTorch/blob/c6b483743598b5f64d520d81e7e5f47ba936d4c9/nets/model_main.py#L55
    """
    def __init__(self,
                 backbone: Union[ResNetFeatureExtractor, UNetFeatureExtractor, ResUNetFeatureExtractor],
                 img_size,
                 task: str,
                 extracted_layers=('conv3', 'conv4', 'conv5'),
                 anchors=(
                         ((64.64, 48.6), (84.24, 106.92), (201.42, 176.04)),
                         ((16.2, 32.94), (33.48, 24.3), (31.86, 64.26)),
                         ((5.4, 7.02), (8.64, 16.2), (17.82, 12.42))
                 ),
                 dataset_stats=None):
        super(YOLOv3WithResNetBackbone, self).__init__()

        if anchors is None:
            assert dataset_stats is not None and 'anchors' in dataset_stats
            anchors = dataset_stats['anchors']

        self.img_size = img_size
        self.anchors = anchors

        if task == 'rsna_pneunomia_detection':
            self.class_names = ['opacity']
        elif task == 'NIH_CXR_pathology_detection':
            self.class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate',
                                 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
        elif task == 'object_cxr':
            self.class_names = ['foreign_object']
        else:
            raise ValueError(task)
        self.num_classes = len(self.class_names)
        assert isinstance(backbone, (ResNetFeatureExtractor, ResUNetFeatureExtractor, UNetFeatureExtractor))
        if isinstance(backbone, ResUNetFeatureExtractor) \
                and all(layer in ResNetFeatureExtractor.LAYERS for layer in extracted_layers):
            backbone = backbone.backbone  # only use the ResNet (downsampling) part of the ResUNet for increased speed
        backbone.set_extracted_feature_layers(extracted_layers)
        self.backbone = backbone
        self.extracted_layers = extracted_layers

        _out_filters = [self.backbone.d[layer] for layer in extracted_layers]

        #  embedding0
        final_out_filter0 = len(anchors[0]) * (5 + self.num_classes)
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(anchors[1]) * (5 + self.num_classes)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(anchors[2]) * (5 + self.num_classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

        self.losses = nn.ModuleList([YOLOv3Loss(scale_anchors, self.num_classes, self.img_size) for scale_anchors in anchors])

    def non_backbone_params(self):
        return chain(self.embedding0.parameters(),
                     self.embedding1.parameters(), self.embedding1_upsample.parameters(), self.embedding1_cbl.parameters(),
                     self.embedding2.parameters(), self.embedding2_upsample.parameters(), self.embedding2_cbl.parameters())

    def backbone_params(self):
        return self.backbone.parameters()

    def _make_cbl(self, _in, _out, ks):
        '''
        cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def forward(self, scan: torch.Tensor, detection_targets=None, frozen_backbone=False, return_predictions=False, **kwargs):
        # (B x H_0 x W_0 x C_0), (B x H_1 x W_1 x C_1), (B x H_2 x W_2 x C_2)
        if frozen_backbone:
            with torch.no_grad():
                x0, x1, x2 = self.extract_features(scan)
        else:
            x0, x1, x2 = self.extract_features(scan)

        # (B x H_0 x W_0 x (anch_0 * (5 + num_cl)))
        # (B x H_1 x W_1 x (anch_1 * (5 + num_cl)))
        # (B x H_2 x W_2 x (anch_2 * (5 + num_cl)))
        out0, out1, out2 = self.apply_predictors(x0, x1, x2)

        with autocast(enabled=False):
            output_0, loss_0, detailed_losses_0 = self.losses[0](out0.float(), detection_targets, return_predictions=return_predictions)
            output_1, loss_1, detailed_losses_1 = self.losses[1](out1.float(), detection_targets, return_predictions=return_predictions)
            output_2, loss_2, detailed_losses_2 = self.losses[2](out2.float(), detection_targets, return_predictions=return_predictions)

        loss = loss_0 + loss_1 + loss_2
        detailed_losses = {
            name: detailed_losses_0[name] + detailed_losses_1[name] + detailed_losses_2[name]
            for name in detailed_losses_0.keys()
        }
        detailed_losses['loss_scale_0'] = loss_0.detach()
        detailed_losses['loss_scale_1'] = loss_1.detach()
        detailed_losses['loss_scale_2'] = loss_2.detach()

        if return_predictions:
            with torch.no_grad():
                output = torch.cat([output_0, output_1, output_2], 1)
                output = non_max_suppression(output, self.num_classes, conf_thres=0.2)
        else:
            output = None

        return output, detection_targets, loss, detailed_losses

    def extract_features(self, scan):
        # extracted maps at 3 scales
        extracted_features = self.backbone(scan)
        x2, x1, x0 = [extracted_features[feature] for feature in self.extracted_layers]
        return x0, x1, x2

    def apply_predictors(self, x0, x1, x2):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        return out0, out1, out2


class YOLOv3Loss(nn.Module):
    """
    Taken from https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/nets/yolo_loss.py
    """
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOv3Loss, self).__init__()
        self.register_buffer('anchors', torch.tensor(anchors, dtype=torch.float))
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_height, self.image_width = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None, return_predictions=False):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.image_height / in_h
        stride_w = self.image_width / in_w
        scaled_anchors = self.anchors.clone()
        scaled_anchors[:, 0] /= stride_w
        scaled_anchors[:, 1] /= stride_h

        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        if targets is not None:
            #  build target
            with torch.no_grad():
                mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_targets(targets, scaled_anchors,
                                                                                 in_w, in_h,
                                                                                 self.ignore_threshold)
            #  losses.
            loss_x = self.bce_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + \
                0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            if self.num_classes == 1:
                loss_cls = loss_conf.new_tensor(0)
            else:
                loss_cls = self.bce_loss(pred_cls * mask.unsqueeze(-1), tcls * mask.unsqueeze(-1))

            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            detailed_losses = {
                "x": loss_x.detach(), "y": loss_y.detach(),
                "w": loss_w.detach(), "h": loss_h.detach(),
                "conf": loss_conf.detach(), "cls": loss_cls.detach()
            }
        if targets is None or return_predictions:
            with torch.no_grad():
                FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
                # Calculate offsets for each grid
                grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                    bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
                grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
                    bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
                # Calculate anchor w, h
                anchor_w = scaled_anchors[:, 0].view(-1, 1)  #.index_select(1, LongTensor([0]))
                anchor_h = scaled_anchors[:, 1].view(-1, 1)  #.index_select(1, LongTensor([1]))
                anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
                anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
                # Add offset and scale with anchors
                pred_boxes = FloatTensor(prediction[..., :4].shape)
                pred_boxes[..., 0] = x.data + grid_x
                pred_boxes[..., 1] = y.data + grid_y
                pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
                pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
                # Results
                _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
                output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                    conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
                output = output.detach()

        if targets is None:
            return output, None, None
        elif return_predictions:
            return output, loss, detailed_losses
        else:
            return None, loss, detailed_losses

    def get_targets(self, targets: List[Dict[str, torch.Tensor]], anchors, in_w, in_h, ignore_threshold):
        """

        :param targets: list of dicts
            boxes: (num_targets x 4), bbox with (x1, y1, w, h)
            classes: (num_targets)
        :param anchors:
        :param in_w:
        :param in_h:
        :param ignore_threshold:
        :return:
        """
        bs = len(targets)
        assert bs > 0

        device = targets[0]['boxes'].device

        # Get shape of anchor box
        anchor_boxes = torch.zeros(self.num_anchors, 4, dtype=torch.float, device=device)  # (num_anchors x 4)
        anchor_boxes[:, 2:] = anchors

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, device=device)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, device=device)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, device=device)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, device=device)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, device=device)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, device=device)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, device=device)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, device=device)

        box_rescale_x = (float(in_w) / float(self.image_width))
        box_rescale_y = (float(in_h) / float(self.image_height))

        for b, target in enumerate(targets):
            boxes = target['boxes']  # bboxes with x1, y1, width, height
            classes = target['classes'].int()

            num_targets = boxes.size()[0]
            assert boxes.size() == (num_targets, 4)
            assert classes.ndim == 1 and classes.size()[0] == num_targets

            # Convert to position relative to box
            gxs = boxes[:, 0] * box_rescale_x  # (T)
            gys = boxes[:, 1] * box_rescale_y  # (T)
            gws = boxes[:, 2] * box_rescale_x  # (T)
            ghs = boxes[:, 3] * box_rescale_y  # (T)
            # convert bbox (with left uppder coords) to center pos + size
            gxs = gxs + gws / 2.  # (T)
            gys = gys + ghs / 2.  # (T)
            # Get grid box indices
            # this is clamped
            gis = gxs.int().clamp_max(in_w - 1)  # (T)
            gjs = gys.int().clamp_max(in_h - 1)  # (T)

            local_gxs = gxs - gis  # (T), 0-1
            local_gys = gys - gjs  # (T), 0-1

            # Get shape of gt box
            gt_boxes = torch.zeros(num_targets, 1, 4, dtype=torch.float, device=device)
            gt_boxes[:, 0, 2] = gws
            gt_boxes[:, 0, 3] = ghs

            for gt_box, gi, gj, local_gx, local_gy, gw, gh, cls \
                    in zip(gt_boxes, gis, gjs, local_gxs, local_gys, gws, ghs, classes):
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_boxes)  # (num_anchors)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = torch.argmax(anch_ious)

                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = local_gx
                ty[b, best_n, gj, gi] = local_gy
                # Width and height
                tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][1] + 1e-16)
                # object
                tconf[b, best_n, gj, gi] = 1

                if self.num_classes == 1:
                    tcls[b, best_n, gj, gi, 0] = 1
                else:
                    # One-hot encoding of label
                    tcls[b, best_n, gj, gi, cls] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    :param prediction (B x num_predictions x prediction)
       prediction:
          0-3: x, y, w, h
          4: confidence
          5-...: class probs

    Returns detections with shape:
        list (length = B) of
        (num_detections_in_img x 7) where 7 consists of x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    B, num_predictions, _ = prediction.size()

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(B, num_predictions, 4)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    outputs = []

    for image_pred in prediction:
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()  # (num_predictions)
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if image_pred.size()[0] == 0:
            outputs.append(None)
            continue
        img_outputs = []
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).detach()
            # Add max detections to outputs
            img_outputs.append(max_detections)
        outputs.append(torch.cat(img_outputs, dim=0))   # (num_detections x 7)

    dict_outputs = []
    for output in outputs:
        if output is None:
            dict_outputs.append({
                'boxes': torch.zeros(0, 4),
                'conf': torch.zeros(0),
                'cls_conf': torch.zeros(0),
                'classes': torch.zeros(0)
            })
            continue
        num_predictions = output.size()[0]

        box_coords = output[:, :4]
        # (x1y1x2y2 to x1y1wh)
        bbox = box_coords.new(*box_coords.size())
        bbox[:, :2] = box_coords[:, :2]
        bbox[:, 2:] = box_coords[:, 2:] - box_coords[:, :2]

        if num_classes == 1:
            cls_conf = output.new_ones(num_predictions, dtype=torch.float)
            classes = output.new_zeros(num_predictions, dtype=torch.long)
        else:
            cls_conf = output[:, 5]
            classes = output[:, 6]

        dict_outputs.append({
            'boxes': bbox,
            'conf': output[:, 4],
            'cls_conf': cls_conf,
            'classes': classes
        })

    return dict_outputs
