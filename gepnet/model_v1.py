"""
"""
from fastai.vision import nn, relu, init_default, AdaptiveConcatPool2d # NormType
from collections import namedtuple
from gepnet.utils import *

arch_config = namedtuple('arch_config', ['comp_graph', 'channels', 'repeat_list', 'classes'])
arch_config.__new__.__defaults__ = (None,) * len(arch_config._fields)


class Layer(nn.Module):
    """Class that is used to build a layer"""
    def __init__(self, cin, comp_graph):
        super(Layer, self).__init__()
        self.inputs = comp_graph[0]
        self.conv_ops = comp_graph[1]
        self.graph_expr = comp_graph[2]

        for op in self.conv_ops:
            if get_op_head(op) == 'conv1x1':
                self.add_module(op, conv2d(cin, ksize=1))
            elif get_op_head(op) == 'conv3x3':
                self.add_module(op, conv2d(cin, ksize=3))
            elif get_op_head(op) == 'dwconv3x3':
                self.add_module(op, conv2d(cin, ksize=3, groups=cin))
            elif get_op_head(op) == 'conv1x3':
                self.add_module(op, conv2d(cin, ksize=(1, 3), padding=(0, 1)))
            elif get_op_head(op) == 'conv3x1':
                self.add_module(op, conv2d(cin, ksize=(3, 1), padding=(1, 0)))
            elif get_op_head(op) == 'maxpool3x3':
                self.add_module(op, pool(pool_type='max'))
            elif get_op_head(op) == 'avgpool3x3':
                self.add_module(op, pool(pool_type='avg'))
            elif get_op_head(op) == 'sepconv3x3':
                self.add_module(op, sepconv2d(cin, ksize=3))
            elif get_op_head(op) == 'sepconv5x5':
                self.add_module(op, sepconv2d(cin, ksize=5))
            elif get_op_head(op) == 'dilconv3x3':
                self.add_module(op, dilconv2d(cin, ksize=3, padding=2, dilation=2))
            elif get_op_head(op) == 'dilconv5x5':
                self.add_module(op, dilconv2d(cin, ksize=5, padding=4, dilation=2))
            else:
                raise NotImplementedError('Unimplemented convolution operation: ', op)

    def forward(self, x):
        return eval(str(self.graph_expr))


class Cell(nn.Module):
    "Class that is uesd to build a cell"
    def __init__(self, cin, comp_graph):
        super(Cell, self).__init__()
        self.n_branch = len(comp_graph)
        self.relu = relu(True)
        for i in range(self.n_branch):
            setattr(self, 'branch_%d' % i, Layer(cin, comp_graph[i]))
        self.convproj = conv2d(cin*self.n_branch, cin, ksize=1, use_relu=False)

    def forward(self, x):
        results = [None] * self.n_branch
        for i in range(self.n_branch):
            results[i] = getattr(self, 'branch_%d' % i)(x)
        results = self.convproj(concat(*results))
        return self.relu(results + x)


class Network(nn.Module):
    """Class that is used to build the entire architecture"""
    def __init__(self, model_config):
        super(Network, self).__init__()
        self.comp_graph = model_config.comp_graph
        self.channels = model_config.channels
        self.repeat_list = model_config.repeat_list
        self.classes = model_config.classes

        self.stem_ = self.stem()
        self.blocks_ = self.blocks()
        self.head_ = self.head()

    def stem(self):
        return nn.Sequential(init_default(nn.Conv2d(3, self.channels, 3, padding=1, bias=False),
                                          nn.init.kaiming_normal_),
                             nn.BatchNorm2d(self.channels))

    def blocks(self):
        assert len(self.comp_graph) <= 4, 'Genes in a chromosome must be <= 4'
        blocks = []
        blk_size = len(self.repeat_list)
        for blk in range(blk_size):
            cin = self.channels
            for cell in range(self.repeat_list[blk]):
                blocks.append(Cell(cin, self.comp_graph))
            if blk < blk_size - 1:
                cout = cin * 2
                blocks.append(conv2dpool(cin, cout, pool_type='max'))
                self.channels = cout
        return nn.Sequential(*blocks)

    def head(self):
        cin = self.channels #* 2
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), #AdaptiveConcatPool2d(1),
                             # nn.Dropout2d(0.5),
                             Flatten(),
                             init_default(nn.Linear(cin, self.classes), nn.init.kaiming_normal_))

    def forward(self, x):
        x = self.stem_(x)
        x = self.blocks_(x)
        x = self.head_(x)
        return x


def get_net(model_config):
    return Network(model_config)

# functions exported
__all__ = ['get_net', 'arch_config']
