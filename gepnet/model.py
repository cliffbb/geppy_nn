"""
"""
from fastai.vision import nn, init_default, AdaptiveConcatPool2d #, NormType
from collections import namedtuple
from gepnet.utils import *

arch_config = namedtuple('arch_config', ['comp_graphs', 'depth_coeff', 'width_coeff', 'channels',
                                         'strides', 'repeats', 'dropout', 'classes'])
arch_config.__new__.__defaults__ = (None,) * len(arch_config._fields)


class GepNetLayer(nn.Module):
    """Class for building GepNet layer"""
    def __init__(self, cin, comp_graph):
        super(GepNetLayer, self).__init__()
        self.output_fan_ins = comp_graph[0][0]
        self.conv_ops = comp_graph[0][1:]
        self.graph_expr = comp_graph[1]
        self.identity = Identity()

        for op in self.conv_ops:
            if get_op_head(op) == 'identity':
                self.add_module(op, Identity())
            elif get_op_head(op) == 'conv1x1':
                self.add_module(op, reluconvbn(cin, ksize=1))
            elif get_op_head(op) == 'conv3x3':
                self.add_module(op, reluconvbn(cin, ksize=3))
            elif get_op_head(op) == 'conv5x5':
                self.add_module(op, reluconvbn(cin, ksize=5))
            elif get_op_head(op) == 'sepconv3x3':
                self.add_module(op, sepconv(cin, ksize=3))
            elif get_op_head(op) == 'sepconv5x5':
                self.add_module(op, sepconv(cin, ksize=5))
            elif get_op_head(op) == 'sepconv7x7':
                self.add_module(op, sepconv(cin, ksize=7))
            elif get_op_head(op) == 'dilsepconv3x3':
                self.add_module(op, dilsepconv(cin, ksize=3, padding=2, dilation=2))
            elif get_op_head(op) == 'dilsepconv5x5':
                self.add_module(op, dilsepconv(cin, ksize=5, padding=4, dilation=2))
            elif get_op_head(op) == 'conv1x3_3x1':
                self.add_module(op, dualconv(cin, ksize=3))
            elif get_op_head(op) == 'conv1x7_7x1':
                self.add_module(op, dualconv(cin, ksize=7))
            elif get_op_head(op) == 'max_pool':
                self.add_module(op, pool(pool_type='max'))
            elif get_op_head(op) == 'avg_pool':
                self.add_module(op, pool(pool_type='avg'))
            else:
                raise NotImplementedError('Unimplemented convolution operation: ', op)

    def forward(self, x):
        return eval(str(self.graph_expr)) + self.identity(x)


class GepNet(nn.Module):
    """Class that is used to build the GepNet entire architecture"""
    def __init__(self, model_config):
        super(GepNet, self).__init__()
        self.comp_graphs = model_config.comp_graphs
        self.depth_coeff = model_config. depth_coeff
        self.width_coeff = model_config.width_coeff
        self.channels = scale_channels(model_config.channels, width_coeff=self.width_coeff)
        self.strides = model_config.strides
        self.repeats = [scale_layer(n, self.depth_coeff) for n in model_config.repeats]
        self.dropout = model_config.dropout
        self.classes = model_config.classes
        #self.n_blocks = len(self.comp_graphs)

        self.stem = self.stem_layer()
        self.blocks = self.gepnet_blocks()
        self.head = self.head_layer()

    def stem_layer(self):
        #stem_multiplier = 2
        channels = self.channels
        stem = convbn(cin=3, cout=channels, ksize=3)
        #self.channels = channels
        return stem

    def gepnet_blocks(self):
        blocks = []
        for n, comp_graph in enumerate(self.comp_graphs):
            channels = self.channels
            repeat = self.repeats[n]
            stride = self.strides[n]
            #n_ops = len(comp_graph[0][0])

            if stride == 1:
                proj_ch = channels
                blocks.append(reluconvbn(cin=channels, cout=proj_ch, ksize=1))
            else:
                proj_ch = channels * 2
                blocks.append(convpool(channels, proj_ch, pool_type='avg'))

            while repeat > 0:
                blocks.append(GepNetLayer(proj_ch, comp_graph))
                repeat -= 1

            self.channels = proj_ch
        return nn.Sequential(*blocks)

    def head_layer(self):
        channels = self.channels * 2
        return nn.Sequential(AdaptiveConcatPool2d(1), #nn.AdaptiveAvgPool2d(1),
                             Flatten(),
                             #nn.Dropout2d(self.dropout),
                             init_default(nn.Linear(channels, self.classes), nn.init.kaiming_normal_))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def get_gepnet(model_config):
    return GepNet(model_config)

# functions exported
__all__ = ['get_gepnet', 'arch_config']
