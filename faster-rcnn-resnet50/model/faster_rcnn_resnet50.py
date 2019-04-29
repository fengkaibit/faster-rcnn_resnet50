from __future__ import absolute_import
import torch
from torchvision.models import resnet50
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils.config import opt
from utils import array_tool
from roi_align.functions.roi_align import RoIAlignFunction


def set_bn_fix(m):   #冻结batchnorm的参数
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad = False

def decom_resnet50():
    model = resnet50(not opt.load_path)
    features_base = torch.nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3)
    features_top = torch.nn.Sequential(
        model.layer4, model.avgpool)
    classifier = torch.nn.Sequential(
        model.fc)

    for layer in features_base[:5]:     #冻结conv1 和conv2_x
        for p in layer.parameters():
            p.requires_grad = False

    features_base.apply(set_bn_fix)
    features_top.apply(set_bn_fix)

    return features_base, features_top, classifier

class FasterRCNNResNet50(FasterRCNN):
    feat_stride = 16

    def __init__(self,
                 n_fg_class= opt.class_num,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        features_base, features_top, classifier = decom_resnet50()

        rpn = RegionProposalNetwork(
            1024, 1024,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = Resnet50RoIHead(
            n_class=n_fg_class + 1,
            roi_size=14,
            spatial_scale=(1. / self.feat_stride),
            features_top=features_top,
            classifier=classifier
        )
        super(FasterRCNNResNet50, self).__init__(
            features_base,
            rpn,
            head,
        )

class Resnet50RoIHead(torch.nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, features_top, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.features_top = features_top
        self.classifier = classifier
        self.cls_loc = torch.nn.Linear(1000, n_class * 4)
        self.score = torch.nn.Linear(1000, n_class)
        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        #self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        self.roi_align = RoIAlignFunction(self.roi_size, self.roi_size, self.spatial_scale)


    def forward(self, x, rois, roi_indices):
        roi_indices = array_tool.totensor(roi_indices).float()
        rois = array_tool.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]  # yx->xy
        indices_and_rois = xy_indices_and_rois.contiguous()  # 把tensor变成在内存中连续分布的形式

        #pool = self.roi(x, indices_and_rois)   #将conv4_x通过roi_pooling
        pool = self.roi_align(x, indices_and_rois)  #将conv4_x通过roi_align
        conv5_out = self.features_top(pool)    #通过conv5_x
        fc_in = conv5_out.view(conv5_out.size(0), -1)
        fc = self.classifier(fc_in)
        roi_cls_locs = self.cls_loc(fc)  # （1000->84）每一类坐标回归
        roi_scores = self.score(fc)  # （1000->21） 每一类类别预测
        return roi_cls_locs, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)   #截断产生正态分布
    else:
        m.weight.data.normal_(mean, stddev)   #普通产生正态分布
        m.bias.data.zero_()