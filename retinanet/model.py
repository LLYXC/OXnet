from torch.nn import functional as F
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from torch.autograd import Function
from retinanet import losses

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
#    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs
#        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P2_x, P3_x, P4_x, P5_x, P6_x, P7_x]
#        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)


        return out2


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, pure_retina=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [
                self.layer1[layers[0] - 1].conv2.out_channels,
                self.layer2[layers[1] - 1].conv2.out_channels,
                self.layer3[layers[2] - 1].conv2.out_channels,
                self.layer4[layers[3] - 1].conv2.out_channels
            ]
        elif block == Bottleneck:
            fpn_sizes = [
                self.layer1[layers[0] - 1].conv3.out_channels,
                self.layer2[layers[1] - 1].conv3.out_channels,
                self.layer3[layers[2] - 1].conv3.out_channels,
                self.layer4[layers[3] - 1].conv3.out_channels
            ]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

        if not pure_retina:
            self.global_classifier = nn.Conv2d(fpn_sizes[3], num_classes, kernel_size=1, stride=1, bias=False)

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

#        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        if not pure_retina:
            # important to avoid gradient explode at the beginning
            self.global_classifier.weight.data.fill_(0)

        #TODO: try removing freeze_bn later
        self.freeze_bn()

        self.num_classes = num_classes
        self.prototypes = torch.zeros(num_classes, 2048, requires_grad=False).float().cuda()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _get_local_map(self, local_maps, reshap_size, normalize=False, detach=False):
        '''Obtain, normalize, and return local attention maps'''
        batch_size = local_maps[0].shape[0]
        attentions = []

        for i in range(batch_size):
            lmaps = [l[i] for l in local_maps]

            # for each class
            attention = []
            for j in range(18):
                target_index = j
                max_val = -1
                # get local map according to maximum of each level
                lmap = None
                for k, lmp in enumerate(lmaps):
                    # take average over anchor predictions
                    # output = torch.mean(output, 3)
                    # take max over anchor predictions
                    lmp = torch.max(lmp, 2)[0]
                    lmp = lmp[..., target_index]
                    if torch.max(lmp) > max_val:
                        max_val = torch.max(lmp)
                        lmap = lmp

                lmap = F.interpolate(lmap.unsqueeze(0).unsqueeze(0), size=reshap_size,
                                     mode='bilinear', align_corners=True)
                lmap = lmap.squeeze().squeeze()

                if normalize:
                    lmap = lmap / lmap.sum()

                if detach:
                    lmap = lmap.detach()

                attention.append(lmap)

            attentions.append(torch.stack(attention, dim=0))

        attentions = torch.stack(attentions, dim=0)

        return attentions

    def update_prototypes(self, cls_features, annotations, probs, decay=0.7):
        # get one hot labels out of bounding box annotations
        batch_size, feat_size = cls_features.shape[0], cls_features.shape[1]
        one_hot = torch.zeros(batch_size, self.num_classes)
        for i in range(batch_size):
            target = annotations[i, :, 4].long()
            for j in range(len(target)):
                if target[j] != -1:
                    one_hot[i, target[j]] = 1
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()

        # update the prototypes
        cls_features = cls_features.contiguous().view(batch_size, self.num_classes, feat_size)
        cls_features = cls_features * one_hot.unsqueeze(2).repeat(1, 1, 2048)
        avg_cls_features = cls_features.sum(dim=0) / (one_hot.sum(dim=0).unsqueeze(1) + 1e-10)

        # if a prototype is updated for the first time, make it the same with the feature, without decay
        decay = torch.ones(self.num_classes, 1).float().cuda() * decay
        neg_prototype_idx = torch.where(self.prototypes.sum(dim=1) == 0)
        decay[neg_prototype_idx] = 0.

        probs = probs.unsqueeze(2)
        weighted_cls_features = (cls_features * probs).sum(dim=0) / (one_hot.unsqueeze(2) * probs + 1e-10).sum(dim=0)
        self.prototypes = decay * self.prototypes + (1. - decay) * weighted_cls_features

        return avg_cls_features, self.prototypes

    def forward(self, inputs, ema=False, pure_retina=False):
        if self.training:
            if pure_retina:
                img_batch, annotations = inputs
            else:
                img_batch, annotations, steps = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x1, x2, x3, x4])
#        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        # local attention maps will be used later
        class_maps = [self.classificationModel(feature) for feature in features]

        classifications = [c.contiguous().view(c.shape[0], -1, self.num_classes) for c in class_maps]
        classification = torch.cat(classifications, dim=1)

        anchors = self.anchors(img_batch)

        # return for retinanet
        if pure_retina:
            if self.training:
                return classification, regression, anchors, annotations
            else:
                transformed_anchors = self.regressBoxes(anchors, regression)
                transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

                scores = torch.max(classification, dim=2, keepdim=True)[0]

                scores_over_thresh = (scores > 0.05)[0, :, 0]

                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just return
                    return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

                classification = classification[:, scores_over_thresh, :]
                transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                scores = scores[:, scores_over_thresh, :]

                anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.5)

                nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

                return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

        # return for oxnet
        else:

            global_feature = F.relu(x4, inplace=False)
            # GAP output
            #global_out = F.adaptive_avg_pool2d(global_maps, (1, 1)).view(global_maps.size(0), -1)

            global_class_maps = self.global_classifier(global_feature)
            attentions = self._get_local_map(class_maps, global_class_maps.shape[-2:], normalize=True, detach=False)
            global_maps = global_class_maps * attentions
            global_out = global_maps.sum(dim=(2, 3))

            cls_features = []
            for i in range(self.num_classes):
                cls_feat = global_feature * attentions[:, i, ...].unsqueeze(1)
                cls_feat = cls_feat.sum(dim=(2, 3))
                cls_features.append(cls_feat)
            cls_features = torch.stack(cls_features, dim=2)

            if self.training:
                if steps == 0:
                    cls_features, prototypes = self.update_prototypes(cls_features, annotations, torch.sigmoid(global_out).detach(), decay=0.)
                else:
                    cls_features, prototypes = self.update_prototypes(cls_features, annotations, torch.sigmoid(global_out).detach(), decay=0.7)
                return classification, regression, anchors, annotations, global_out, cls_features, prototypes

            else:
                if ema:
                    _classification, _regression = classification.clone(), regression.clone()
                    res = {}
                    for k in range(_classification.shape[0]):
                        classification, regression = torch.unsqueeze(_classification[k,:,:], dim=0), torch.unsqueeze(_regression[k,:,:], dim=0)

                        transformed_anchors = self.regressBoxes(anchors, regression)
                        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

                        scores = torch.max(classification, dim=2, keepdim=True)[0]

                        scores_over_thresh = (scores > 0.01)[0, :, 0]

                        if scores_over_thresh.sum() == 0:
                            res[k] = {'ema_bbox_scores': torch.zeros(0).cuda(),
                                      'ema_classes'    : torch.zeros(0).cuda(),
                                      'ema_bboxes'     : torch.zeros(0, 4).cuda(),
                                      'global_out'     : global_out[k]}
                        else:
                            classification = classification[:, scores_over_thresh, :]
                            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                            scores = scores[:, scores_over_thresh, :]

                            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

                            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

                            idxs             = torch.where(nms_scores > 0.5)
                            ema_bbox_scores  = nms_scores[idxs[0]]
                            ema_classes      = nms_class[idxs[0]].float()
                            ema_bboxes = transformed_anchors[0, :, :][idxs[0], :]

                            res[k] = {'ema_bbox_scores' : ema_bbox_scores,
                                      'ema_classes'     : ema_classes,
                                      'ema_bboxes'      : ema_bboxes,
                                      'global_out'      : global_out[k]}

                    return _classification, res, global_out

                # when not in ema mode, using batch size = 1 is enough
                else:
                    transformed_anchors = self.regressBoxes(anchors, regression)
                    transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

                    scores = torch.max(classification, dim=2, keepdim=True)[0]

                    scores_over_thresh = (scores > 0.05)[0, :, 0]

                    if scores_over_thresh.sum() == 0:
                        # no boxes to NMS, just return
                        return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4), global_out]

                    classification = classification[:, scores_over_thresh, :]
                    transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
                    scores = scores[:, scores_over_thresh, :]

                    anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.5)

                    nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

                    return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :], global_out]

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _maxp_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        threshold = threshold * 0.7
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
