import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the classification loss
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the regression loss
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

class ConsistencyLoss(nn.Module):

    def forward(self, classifications, regressions, anchors,
                ema_classifications, ema_res, annotations):
        alpha_0 = 0.05
        alpha_1 = 0.95
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_consistency_losses = []
        regression_consistency_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        if torch.cuda.is_available():
            annotations = annotations.cuda()

        for j in range(batch_size):

            classification     = classifications[j, :, :]
            ema_classification = ema_classifications[j, :, :]

            regression     = regressions[j, :, :]
            ema_bbox_score = ema_res[j]['ema_bbox_scores']
            ema_class      = ema_res[j]['ema_classes']
            ema_bbox       = ema_res[j]['ema_bboxes']
            global_out     = ema_res[j]['global_out']

            if torch.cuda.is_available():
                ema_class = ema_class.cuda()

            # if there is ema_bbox prediction, filter out those which are not in the image-level GT
            if ema_bbox.shape[0] != 0:
                ema_class_copy = ema_class.clone()
                ema_bbox_copy = ema_bbox.clone()
                # if none of pred classes belong to GT
                if [_pred for _pred in ema_class_copy if _pred in annotations[j, :, 4]] == []:
                    ema_class = torch.zeros(0).cuda()
                    ema_bbox = torch.zeros(0, 4).cuda()
                else:
                    ema_class = torch.stack([_pred for _pred in ema_class_copy if _pred in annotations[j, :, 4]])
                    ema_class = ema_class.cuda()
                    ema_bbox = torch.stack(
                        [_bbox for _bbox, _pred in zip(ema_bbox_copy, ema_class_copy) if _pred in annotations[j, :, 4]])

            if ema_bbox.shape[0] == 0:
                if torch.cuda.is_available():
                    classification_consistency_losses.append(torch.tensor(0).float().cuda())
                    regression_consistency_losses.append(torch.tensor(0).float().cuda())
                else:
                    classification_consistency_losses.append(torch.tensor(0).float().cuda())
                    regression_consistency_losses.append(torch.tensor(0).float())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            ema_classification = torch.clamp(ema_classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], ema_bbox[:, :4])  # num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the consistency loss for classification
            targets = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = ema_classification[torch.lt(IoU_max, 0.4), :]

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()


            assigned_annotations = ema_bbox[IoU_argmax, :]

            targets[positive_indices, :] = ema_classification[positive_indices, :]
            alpha_factor = alpha_0 + ema_classifications*(alpha_1-alpha_0)
            focal_weight = torch.abs(ema_classification-classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_cons_loss = focal_weight * bce
            cls_cons_loss = torch.where(torch.ne(targets, -1.0), cls_cons_loss, torch.zeros(cls_cons_loss.shape).cuda())
            classification_consistency_losses.append(cls_cons_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # Regression loss learns from teacher's output bbox
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]

                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                regression_consistency_losses.append(regression_loss.mean())

            else:
                if torch.cuda.is_available():
                    regression_consistency_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_consistency_losses.append(torch.tensor(0).float())

        return torch.stack(classification_consistency_losses).mean(dim=0, keepdim=True), torch.stack(regression_consistency_losses).mean(dim=0, keepdim=True)

class GlobalClassificationLoss(nn.Module):

    def __init__(self, pos_weight=None, with_logit=True):
        super(GlobalClassificationLoss, self).__init__()
        self.pos_weight = pos_weight
        if pos_weight is not None:
            self.pos_weight = torch.FloatTensor(self.pos_weight).cuda()
        self.with_logit = with_logit

    def forward(self, global_preds, annotations):
        batch_size  = global_preds.shape[0]
        one_hot = torch.zeros(global_preds.shape)

        if not self.with_logit:
            global_preds = torch.clamp(global_preds, 1e-10, 1.-1e-10)

        for i in range(batch_size):
            target = annotations[i, :, 4].long()
            for j in range(len(target)):
                if target[j] != -1:
                    one_hot[i, target[j]] = 1

        if torch.cuda.is_available():
            one_hot = one_hot.cuda()

        if self.with_logit:
            return F.binary_cross_entropy_with_logits(global_preds, one_hot, pos_weight=self.pos_weight).mean()
        else:
            return F.binary_cross_entropy(global_preds, one_hot).mean()


class GlobalContrastiveLoss(nn.Module):
    # Contrastive loss referenced from https://github.com/ChrisAllenMing/GPA-detection/blob/fedcb501558b6aff8eccf09d319e9712c5473bad/lib/model/adaptive_faster_rcnn/faster_rcnn.py

    def __init__(self, num_classes):
        super(GlobalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = 1.

    def distance(self, feat_1, feat_2):
        output = torch.pow(feat_1 - feat_2, 2.0).mean()
        return output

    def forward(self, cls_features, prototypes, annotations, steps):
        prototypes = prototypes.detach()

        # get one hot labels out of bounding box annotations
        batch_size = annotations.shape[0]
        one_hot = torch.zeros(batch_size, self.num_classes)
        for i in range(batch_size):
            target = annotations[i, :, 4].long()
            for j in range(len(target)):
                if target[j] != -1:
                    one_hot[i, target[j]] = 1
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        one_hot = one_hot.sum(dim=0).unsqueeze(1)

        intra_loss = torch.zeros(1).float().cuda()
        intra_loss_count = 0
        inter_loss = torch.zeros(1).float().cuda()
        inter_loss_count = 0
        for i in range(cls_features.shape[0]):
            if one_hot[i]!=0:
                cls_feat_i = cls_features[i]
                for j in range(cls_features.shape[0]):
                    if prototypes[j].sum() != 0:
                        prototype_j = prototypes[j]
                        if i == j:
                            intra_loss += self.distance(cls_feat_i, prototype_j)
                            intra_loss_count += 1
                        else:
                            inter_loss += torch.pow(
                                (self.margin - torch.sqrt(self.distance(cls_feat_i, prototype_j))) / self.margin,
                                2) * torch.pow(
                                torch.max(self.margin - torch.sqrt(self.distance(cls_feat_i, prototype_j)),
                                          torch.tensor(0).float().cuda()), 2.0)
                            inter_loss_count += 1

        intra_loss = intra_loss / (intra_loss_count + 1e-10)
        inter_loss = inter_loss / (inter_loss_count + 1e-10)

        return intra_loss.mean(), inter_loss.mean()

