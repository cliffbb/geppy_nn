"""
"""
from fastai.vision import nn, relu, init_default, AdaptiveConcatPool2d # NormType
from collections import namedtuple
from gepnet.utils import *

arch_config = namedtuple('arch_config', ['comp_graph', 'depth_coeff', 'width_coeff', 'channels',
                                         'repeat_list', 'classes'])
arch_config.__new__.__defaults__ = (None,) * len(arch_config._fields)


class GepNetLayer(nn.Module):
    """Class for building GepNet layer"""
    def __init__(self, cin, comp_graph):
        super(GepNetLayer, self).__init__()
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
                self.add_module(op, pool(pool_type='max', ksize=3))
            else:
                raise NotImplementedError('Unimplemented convolution operation: ', op)

    def forward(self, x):
        return eval(str(self.graph_expr))


class GepBlock(nn.Module):
    def __init__(self, cin, comp_graph):
        super(GepBlock, self).__init__()
        self.paths = len(comp_graph)
        self.relu = relu(True)
        for i in range(self.paths):
            setattr(self, 'path_%d' % i, GepNetLayer(cin, comp_graph[i]))
        self.convproj = conv2d(cin*self.paths, cin, ksize=1, use_relu=False)

    def forward(self, x):
        results = [None] * self.paths
        for i in range(self.paths):
            results[i] = getattr(self, 'path_%d' % i)(x)
        results = self.convproj(concat(*results))
        return self.relu(results + x)


class GepNet(nn.Module):
    """Class that is used to build the GepNet entire architecture"""
    def __init__(self, model_config):
        super(GepNet, self).__init__()
        self.comp_graph = model_config.comp_graph
        self.depth_coeff = model_config. depth_coeff
        self.width_coeff = model_config.width_coeff
        self.channels = scale_channels(model_config.channels, width_coeff=self.width_coeff)
        self.repeat_list = [scale_layer(n, self.depth_coeff) for n in model_config.repeat_list]
        self.classes = model_config.classes

        self.stem = self.stem_layer()
        self.blocks = self.gepnet_blocks()
        self.head = self.head_layer()

    def stem_layer(self):
        stem = conv2d(cin=3, cout=self.channels, ksize=3)
        return stem

    def gepnet_blocks(self):
        assert len(self.comp_graph) <= 4, 'Genes in a chromosome must be <= 4'
        blocks = []
        blk_size = len(self.repeat_list)
        for blk in range(blk_size):
            cin = self.channels
            for cell in range(self.repeat_list[blk]):
                blocks.append(GepBlock(cin, self.comp_graph))
            if blk < blk_size - 1:
                cout = cin * 2
                blocks.append(conv2dpool(cin, cout, pool_type='max'))
                self.channels = cout
        return nn.Sequential(*blocks)

    def head_layer(self):
        cin = self.channels # * 2
        return nn.Sequential(nn.AdaptiveAvgPool2d(1),  #AdaptiveConcatPool2d(1),
                             #nn.Dropout2d(0.5),
                             Flatten(),
                             init_default(nn.Linear(cin, self.classes), nn.init.kaiming_normal_))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def get_gepnet(model_config):
    return GepNet(model_config)

# functions exported
__all__ = ['get_gepnet', 'arch_config']
