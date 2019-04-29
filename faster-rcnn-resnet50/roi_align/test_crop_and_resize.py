import torch
from torch.autograd import Variable
from roi_align.functions.roi_align import RoIAlignFunction

def crop_and_resize(pool_size, feature_map, boxes, box_ind):
    if boxes.shape[1]==5:
        x1, y1, x2, y2, _= boxes.chunk(5, dim=1)
    else:
        x1, y1, x2, y2= boxes.chunk(4, dim=1)
    
    box_ind=box_ind.view(-1,1).float()

    boxes = torch.cat((box_ind, x1, y1, x2, y2), 1)
    return RoIAlignFunction(pool_size[0],pool_size[1], 1)(feature_map, boxes)


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var
