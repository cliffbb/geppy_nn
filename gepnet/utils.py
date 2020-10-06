"""
"""
from fastai.vision.all import nn, init_default, torch, math, BatchNorm, NormType


def pool(pool_type):
    assert pool_type in ['avg', 'max']
    if pool_type == 'max':
        return nn.MaxPool2d(3, stride=1, padding=1)
    if pool_type == 'avg':
        return nn.AvgPool2d(3, stride=1, padding=1, ceil_mode=True, count_include_pad=False)


def conv2dpool(cin, cout, pool_type, bn=NormType.Weight):
    assert pool_type in ['avg', 'max']
    if pool_type == 'max':
        return nn.Sequential(nn.MaxPool2d(2, stride=2),
                             init_default(nn.Conv2d(cin, cout, 1, bias=False), nn.init.kaiming_normal_),
                             BatchNorm(cout, norm_type=bn))
    if pool_type == 'avg':
        return nn.Sequential(nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False),
                             init_default(nn.Conv2d(cin, cout, 1, bias=False), nn.init.kaiming_normal_),
                             BatchNorm(cout, norm_type=bn))


def conv2dtransp(cin, cout, ksize=2, stride=2):
    return init_default(nn.ConvTranspose2d(cin, cout, ksize, stride=stride),
                        nn.init.kaiming_normal_)


# def stem_blk(cin, cout=None, ksize=3, stride=1, bn=NormType.BatchZero, pool='avg'):
#     if cout is None: cout = cin
#     padding = ksize // 2
#     layer = [ConvLayer(cin, cout, ks=ksize, stride=stride, padding=padding, norm_type=bn)]
#     if pool=='max': layer.append(nn.MaxPool2d(2, stride=2))
#     if pool=='avg': layer.append(nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False))
#
#     layer.append(ConvLayer(cout, cout*2, ks=ksize, stride=stride, padding=padding, norm_type=bn))
#     if pool=='max': layer.append(nn.MaxPool2d(2, stride=2))
#     if pool=='avg': layer.append(nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False))
#     return nn.Sequential(*layer)


def stem_blk(cin, cout=None, ksize=3, stride=1, pool='avg', bn=NormType.Weight):
    if cout is None: cout = cin
    padding = ksize // 2
    layer = [init_default(nn.Conv2d(cin, cout, ksize, stride=stride, padding=padding, bias=False),
                          nn.init.kaiming_normal_)]
    layer.append(BatchNorm(cout, norm_type=bn))
    layer.append(nn.ReLU(True))
    if pool=='max': layer.append(nn.MaxPool2d(2, stride=2))
    if pool=='avg': layer.append(nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False))

    layer.append(init_default(nn.Conv2d(cout, cout*2, ksize, stride=stride, padding=padding, bias=False),
                              nn.init.kaiming_normal_))
    layer.append(BatchNorm(cout*2, norm_type=bn))
    layer.append(nn.ReLU(True))
    if pool=='max': layer.append(nn.MaxPool2d(2, stride=2))
    if pool=='avg': layer.append(nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False))
    return nn.Sequential(*layer)


def conv2d(cin, cout=None, ksize=3, stride=1, padding=None, dilation=None, groups=None, bn=NormType.Weight):
    if cout is None: cout = cin
    if padding is None: padding = ksize // 2
    if dilation is None: dilation = 1
    if groups is None: groups = 1
    layer = nn.Sequential(init_default(nn.Conv2d(cin, cout, ksize, stride=stride, padding=padding,
                                                 dilation=dilation, groups=groups, bias=False),
                                       nn.init.kaiming_normal_),
                          BatchNorm(cout, norm_type=bn), nn.ReLU(True))
    return layer
    # layer = ConvLayer(cin, cout, ks=ksize, stride=stride, padding=padding,
    #                   dilation=dilation, groups=groups, norm_type=bn)


def dilconv2d(cin, cout=None, ksize=3, stride=1, padding=2, dilation=2, bn=NormType.Weight):
    if cout is None: cout = cin
    layer = nn.Sequential(init_default(nn.Conv2d(cin, cin, ksize, stride=stride, padding=padding,
                                                 dilation=dilation, groups=cin, bias=False), nn.init.kaiming_normal_),
                          BatchNorm(cin, norm_type=bn), nn.ReLU(True),
                          init_default(nn.Conv2d(cin, cout, 1, padding=0, bias=False), nn.init.kaiming_normal_),
                          BatchNorm(cout, norm_type=bn), nn.ReLU(True))
    return layer


# def dilconv2d(cin, cout=None, ksize=3, stride=1, padding=2, dilation=2, bn=NormType.BatchZero):
#     if cout is None: cout = cin
#     layer = ConvLayer(cin, cout, ks=ksize, stride=stride, padding=padding, dilation=dilation, norm_type=bn)
#     return layer


# def dilconv2d(cin, cout=None, ksize=3, stride=1, padding=2, dilation=2, bn=NormType.BatchZero):
#     if cout is None: cout = cin
#     layer = nn.Sequential(ConvLayer(cin, cout, ks=ksize, stride=stride, padding=padding, dilation=dilation,
#                                     groups=cin, norm_type=bn),
#                           ConvLayer(cin, cin, ks=1, padding=0, groups=1, norm_type=bn))
#     return layer


# def sepconv2d(cin, cout=None, ksize=3, stride=1, padding=None, bn=NormType.BatchZero):
#     if cout is None: cout = cin
#     if padding is None: padding = ksize // 2
#     layer = nn.Sequential(ConvLayer(cin, cin, ks=ksize, stride=stride, padding=padding, groups=cin, norm_type=bn),
#                           ConvLayer(cin, cin, ks=1, padding=0, groups=1, norm_type=bn))
#     return layer


# def sepconv2d(cin, cout=None, ksize=3, stride=1, padding=None, bn=NormType.BatchZero):
#     if cout is None: cout = cin
#     if padding is None: padding = ksize // 2
#     layer = nn.Sequential(ConvLayer(cin, cin, ks=ksize, stride=stride, padding=padding, groups=cin, norm_type=bn),
#                           ConvLayer(cin, cin, ks=1, padding=0, groups=1, norm_type=bn),
#                           ConvLayer(cin, cin, ks=ksize, stride=1, padding=padding, groups=cin, norm_type=bn),
#                           ConvLayer(cin, cout, ks=1, padding=0, groups=1, norm_type=bn))
#     return layer


def sepconv2d(cin, cout=None, ksize=3, stride=1, padding=None, bn=NormType.Weight):
    if cout is None: cout = cin
    if padding is None: padding = ksize // 2
    layer = nn.Sequential(init_default(nn.Conv2d(cin, cin, ksize, stride=stride, padding=padding,
                                                 groups=cin, bias=False), nn.init.kaiming_normal_),
                          init_default(nn.Conv2d(cin, cin, 1, padding=0, bias=False), nn.init.kaiming_normal_),
                          BatchNorm(cin, norm_type=bn), nn.ReLU(True),
                          init_default(nn.Conv2d(cin, cin, ksize, stride=1, padding=padding,
                                                 groups=cin, bias=False), nn.init.kaiming_normal_),
                          init_default(nn.Conv2d(cin, cout, 1, padding=0, bias=False), nn.init.kaiming_normal_),
                          BatchNorm(cin, norm_type=bn), nn.ReLU(True))
    return layer


def scale_channels(ch, depth_div=8, width_coeff=None, min_depth=None):
    """*Ref: EfficientNets (Tan, et. al., 2019)"""
    if not width_coeff: return ch
    ch *= width_coeff
    min_depth = min_depth or depth_div
    new_ch = max(min_depth, int(ch + depth_div / 2) // depth_div * depth_div)
    if new_ch < 0.9 * ch: new_ch += depth_div
    return int(new_ch)


def scale_layer(ops, depth_coeff=None):
    """*Ref: EfficientNets (Tan, et. al., 2019)"""
    if not depth_coeff: return int(ops)
    return int(math.ceil(ops * depth_coeff))


def add(*tensors):
    x = 0
    for t in tensors:
        x += t
    return x


def concat(*tensors):
    return torch.cat(tensors, dim=1)


def get_op_head(op):
    return op[:op.rfind('_')]


def get_op_tail(op):
    return op[-1:]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params/1e6


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# exported functions
__all__ = ['add', 'concat', 'get_op_head', 'get_op_tail', 'scale_channels', 'scale_layer',
           'conv2d', 'stem_blk', 'Flatten', 'pool', 'conv2dpool', 'count_parameters',
           'sepconv2d', 'dilconv2d', 'conv2dtransp']
