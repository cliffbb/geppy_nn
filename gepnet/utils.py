"""
"""
from fastai.vision import nn, init_default, torch, math, relu, batchnorm_2d, NormType


def pool(pool_type):
    assert pool_type in ['avg', 'max']
    if pool_type == 'max':
        return nn.MaxPool2d(2, stride=2)
    if pool_type == 'avg':
        return nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False)


def conv2dpool(cin, cout, pool_type, bn=NormType.Batch):
    assert pool_type in ['avg', 'max']
    if pool_type == 'max':
        return nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                init_default(nn.Conv2d(cin, cout, 1, bias=False), nn.init.kaiming_normal_),
                batchnorm_2d(cout, norm_type=bn))
                #relu(True))
    if pool_type == 'avg':
        return nn.Sequential(
                nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False),
                init_default(nn.Conv2d(cin, cout, 1, bias=False), nn.init.kaiming_normal_),
                batchnorm_2d(cout, norm_type=bn))
                #relu(True))

def stem_blk(cin, cout=None, ksize=3, stride=1, use_relu=True, use_bn=True, bn=NormType.Batch,
             bias=False, pool='avg'):
    if cout is None: cout = cin
    padding = ksize // 2
    layer = [init_default(nn.Conv2d(cin, cout, ksize, stride=stride, padding=padding, bias=bias),
                          nn.init.kaiming_normal_)]
    if use_bn: layer.append(batchnorm_2d(cout, norm_type=bn))
    if use_relu: layer.append(relu(True))
    if pool=='max': layer.append(nn.MaxPool2d(2, stride=2))
    if pool=='avg': layer.append(nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False))

    layer.append(init_default(nn.Conv2d(cout, cout*2, ksize, stride=stride, padding=padding, bias=bias),
                              nn.init.kaiming_normal_))
    if use_bn: layer.append(batchnorm_2d(cout*2, norm_type=bn))
    if use_relu: layer.append(relu(True))
    if pool=='max': layer.append(nn.MaxPool2d(2, stride=2))
    if pool=='avg': layer.append(nn.AvgPool2d(2, stride=2, ceil_mode=True, count_include_pad=False))
    return nn.Sequential(*layer)


def conv2d(cin, cout=None, ksize=None, stride=1, padding=None, dilation=None, groups=None,
                use_relu=True, use_bn=True, bn=NormType.Batch, bias=False):
    if cout is None: cout = cin
    if padding is None: padding = ksize // 2
    if dilation is None: dilation = 1
    if groups is None: groups = 1
    layer = [init_default(nn.Conv2d(cin, cout, ksize, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias), nn.init.kaiming_normal_)]
    if use_bn: layer.append(batchnorm_2d(cout, norm_type=bn))
    if use_relu: layer.append(relu(True))
    return nn.Sequential(*layer)


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
           'conv2d', 'stem_blk', 'Flatten', 'pool', 'conv2dpool', 'count_parameters']
