from __future__ import absolute_import
import torch
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils.config import opt
from utils import array_tool
from roi_align.functions.roi_align import RoIAlignFunction

def decom_vgg16():
    if opt.caffe_pretrain:
        model = vgg16(pretrained= False)
        if not opt.load_path:
            model.load_state_dict(torch.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]  #vgg16前30层为特征提取层
    classifier = model.classifier  #vgg16分类层
    classifier = list(classifier)
    del classifier[6]  #删除最后一层分类层，Linear（4096 -> 1000）
    if not opt.use_dropout:
        del classifier[5]
        del classifier[2]
    classifier = torch.nn.Sequential(*classifier)

    #冻结卷积层的前10层
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return torch.nn.Sequential(*features), classifier  #返回特征层和分类层

class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self,
                 n_fg_class = 20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class= n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )

class VGG16RoIHead(torch.nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = torch.nn.Linear(4096, n_class * 4)
        self.score = torch.nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        #self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        self.roi_align = RoIAlignFunction(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):   #rois (R', 4), roi_indices(R',)
        roi_indices = array_tool.totensor(roi_indices).float()
        rois = array_tool.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]  #yx->xy
        indices_and_rois = xy_indices_and_rois.contiguous() #把tensor变成在内存中连续分布的形式

        #pool = self.roi(x, indices_and_rois)
        pool = self.roi_align(x, indices_and_rois)  # 通过roi_align
        pool = pool.view(pool.size(0), -1)   # (batch_size, CHW拉直)

        fc7 = self.classifier(pool)  #decom_vgg16（）得到的calssifier,得到4096
        roi_cls_locs = self.cls_loc(fc7)  #（4096->84）每一类坐标回归
        roi_scores = self.score(fc7)   #（4096->21） 每一类类别预测
        return roi_cls_locs, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)   #截断产生正态分布
    else:
        m.weight.data.normal_(mean, stddev)   #普通产生正态分布
        m.bias.data.zero_()