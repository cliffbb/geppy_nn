"""
"""
from fastai.vision.all import * #nn, init_default , torch #AdaptiveConcatPool2d # NormType
from collections import namedtuple, OrderedDict
from gepnet.utils import *

arch_config = namedtuple('arch_config', ['comp_graphs', 'channels', 'classes', 'img_size'])
arch_config.__new__.__defaults__ = (None,) * len(arch_config._fields)


class Layer(nn.Module):
    """Class that is used to build a layer"""
    def __init__(self, cin, comp_graph):
        super(Layer, self).__init__()
        self.inputs = comp_graph[0]
        self.conv_ops = comp_graph[1]
        self.graph_expr = comp_graph[2]

        for op in self.conv_ops:
            if get_op_head(op) == 'sepconv3x3':
                self.add_module(op, sepconv2d(cin, ksize=3, double_stack=False))
            elif get_op_head(op) == 'sepconv5x5':
                self.add_module(op, sepconv2d(cin, ksize=5, double_stack=False))
            elif get_op_head(op) == 'dilconv3x3':
                self.add_module(op, dilconv2d(cin, ksize=3, padding=2, dilation=2, seperate=False))
            elif get_op_head(op) == 'dilconv5x5':
                self.add_module(op, dilconv2d(cin, ksize=5, padding=4, dilation=2, seperate=False))
            elif get_op_head(op) == 'maxpool3x3':
                self.add_module(op, pooling(pool_type='max'))
            elif get_op_head(op) == 'avgpool3x3':
                self.add_module(op, pooling(pool_type='avg'))
            else:
                raise NotImplementedError('Unimplemented convolution operation: ', op)

    def forward(self, x):
        return eval(str(self.graph_expr))


class Cell(nn.Module):
    "Class that is uesd to build a cell"
    def __init__(self, cin, comp_graph):
        super(Cell, self).__init__()
        self.n_branch = len(comp_graph)
        self.relu = nn.ReLU(True)
        for i in range(self.n_branch):
            setattr(self, 'branch_%d' % i, Layer(cin, comp_graph[i]))
        # self.convproj = conv2d(cin*self.n_branch, cin, ksize=1)
        self.convproj = conv2d(cin*self.n_branch, cin*2, ksize=3, stride=2)

    def forward(self, x):
        cell = [None] * self.n_branch
        for i in range(self.n_branch):
            cell[i] = getattr(self, 'branch_%d' % i)(x)
        cell = self.convproj(Cat(cell))
        return cell # self.relu(cell + x)


class Network(nn.Module):
    """Class that is used to build the entire architecture"""
    def __init__(self, model_config):
        super(Network, self).__init__()
        # super().__init__()
        self.comp_graphs = model_config.comp_graphs
        self.channels = model_config.channels
        self.classes = model_config.classes
        self.img_size = model_config.img_size

        self.stem_ = self.stem()
        for name, module in self.encoder().named_children():
            setattr(self, name, module)
        for name, module in self.decoder().named_children():
            setattr(self, name, module)
        self.head_ = self.head()

    def stem(self):
        return stem_blk(cin=3, cout=self.channels, ksize=3, stride=2, double_stack=True)

    def encoder(self):
        cell_blks = []
        # cin = self.channels * 2
        n_blk = 4
        pad_dil = 24
        for i in range(n_blk):
            cin = self.channels * 2
            cell_blks.append(('cell_{}'.format(i), Cell(cin, self.comp_graphs)))
            cell_blks.append(('aspp_{}'.format(i), ASPP(cin, padding=pad_dil, dilation=pad_dil,
                                                        n_classes=self.classes)))
            pad_dil /= 2
        return nn.Sequential(OrderedDict(cell_blks))

    # def decoder(self):
    #     decoder_blks = []
    #     n_blk = 4
    #     for i in range(n_blk):
    #         cin = self.channels
    #         if i < n_blk - 1:
    #             cout = cin // 2
    #             n = n_blk - (1 + i)
    #             decoder_blks.append(('upool{}'.format(n), conv2dtransp(cin, cout)))
    #             for c in range(1):
    #                 decoder_blks.append(('dec{}_{}'.format(n,c), Cell(cout, self.comp_graphs)))
    #             self.channels = cout
    #         else:
    #             n = n_blk - (1 + i)
    #             for c in range(1):
    #                 decoder_blks.append(('dec{}_{}'.format(n,c), Cell(self.channels, self.comp_graphs)))
    #     return nn.Sequential(OrderedDict(decoder_blks))

    def head(self):
        cin = self.channels
        return init_default(nn.Conv2d(cin, self.classes, 1), nn.init.kaiming_normal_)

    def forward(self, x):
        stem = self.stem_(x)
        cell_0 = self.cell_0(stem)
        cell_1 = self.cell_1(cell_0)
        cell_2 = self.cell_2(cell_1)
        cell_3 = self.cell_3(cell_2)

        aspp_0 = self.aspp_0(cell_0)
        aspp_1 = self.aspp_1(cell_1)
        aspp_2 = self.aspp_2(cell_2)
        aspp_3 = self.aspp_3(cell_3)

        upsample = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        aspp_0 = upsample(aspp_0)
        aspp_1 = upsample(aspp_1)
        aspp_2 = upsample(aspp_2)
        aspp_3 = upsample(aspp_3)
        feature_map = aspp_0 + aspp_1 + aspp_2 + aspp_3
        return feature_map


# class Network(nn.Module):
#     """Class that is used to build the entire architecture"""
#     def __init__(self, model_config):
#         super(Network, self).__init__()
#         # super().__init__()
#         self.comp_graphs = model_config.comp_graphs
#         self.channels = model_config.channels
#         self.classes = model_config.classes
#         self.stem_ = self.stem()
#         for name, module in self.encoder().named_children():
#             setattr(self, name, module)
#         for name, module in self.decoder().named_children():
#             setattr(self, name, module)
#         self.head_ = self.head()
#
#     def stem(self):
#         return stem_blk(cin=3, cout=self.channels, ksize=3, stride=2, double_stack=True)
#
#     def encoder(self):
#         encoder_blks = []
#         n_blk = 4
#         for i in range(n_blk):
#             cin = self.channels
#             for c in range(1):
#                 encoder_blks.append(('enc{}_{}'.format(i,c), Cell(cin, self.comp_graphs)))
#             if i < n_blk - 1:
#                 cout = cin * 2
#                 encoder_blks.append(('dpool{}'.format(i), conv2dpool(cin, cout, pool_type='max')))
#                 self.channels = cout
#         return nn.Sequential(OrderedDict(encoder_blks))
#
#     def decoder(self):
#         decoder_blks = []
#         n_blk = 4
#         for i in range(n_blk):
#             cin = self.channels
#             if i < n_blk - 1:
#                 cout = cin // 2
#                 n = n_blk - (1 + i)
#                 decoder_blks.append(('upool{}'.format(n), conv2dtransp(cin, cout)))
#                 for c in range(1):
#                     decoder_blks.append(('dec{}_{}'.format(n,c), Cell(cout, self.comp_graphs)))
#                 self.channels = cout
#             else:
#                 n = n_blk - (1 + i)
#                 for c in range(1):
#                     decoder_blks.append(('dec{}_{}'.format(n,c), Cell(self.channels, self.comp_graphs)))
#         return nn.Sequential(OrderedDict(decoder_blks))
#
#     def head(self):
#         cin = self.channels
#         return init_default(nn.Conv2d(cin, self.classes, 1), nn.init.kaiming_normal_)
#
#     def forward(self, x):
#         stem = self.stem_(x)
#         # enc0 = self.enc0_1(self.enc0_0(stem))
#         # enc1 = self.enc1_1(self.enc1_0(self.dpool0(enc0)))
#         # enc2 = self.enc2_1(self.enc2_0(self.dpool1(enc1)))
#         # enc3 = self.enc3_1(self.enc3_0(self.dpool2(enc2)))
#         enc0 = self.enc0_0(stem)
#         enc1 = self.enc1_0(self.dpool0(enc0))
#         enc2 = self.enc2_0(self.dpool1(enc1))
#         enc3 = self.enc3_0(self.dpool2(enc2))
#
#         upool3 = self.upool3(enc3)
#         # dec3 = self.dec3_1(self.dec3_0(upool3 + enc2))
#         # upool2 = self.upool2(dec3)
#         # dec2 = self.dec2_1(self.dec2_0(upool2 + enc1))
#         # upool1 = self.upool1(dec2)
#         # dec1 = self.dec1_1(self.dec1_0(upool1 + enc0))
#         # dec0 = self.dec0_1(self.dec0_0(dec1 + stem))
#         dec3 = self.dec3_0(upool3 + enc2)
#         upool2 = self.upool2(dec3)
#         dec2 = self.dec2_0(upool2 + enc1)
#         upool1 = self.upool1(dec2)
#         dec1 = self.dec1_0(upool1 + enc0)
#         dec0 = self.dec0_0(dec1 + stem)
#         return self.head_(dec0)


# class Network(nn.Sequential):
#     """Class that is used to build the entire architecture"""
#     def __init__(self, model_config):
#         super(Network, self).__init__()
#         self.comp_graphs = model_config.comp_graphs
#         self.channels = model_config.channels
#         self.classes = model_config.classes
#         #self.softmax = nn.Softmax2d()
#         self.stem_ = self.stem()
#         self.head_ = self.head()
#         for name, module in self.encoder().named_children():
#             setattr(self, name, module)
#         for name, module in self.decoder().named_children():
#             setattr(self, name, module)
#
#     def stem(self):
#         return nn.Sequential(init_default(nn.Conv2d(3, self.channels, 3, padding=1, bias=False),
#                                           nn.init.kaiming_normal_),
#                              nn.BatchNorm2d(self.channels))
#
#     def encoder(self):
#         encoder_blks = []
#         n_blk = 4
#         for i in range(n_blk):
#             cin = self.channels
#             for c in range(1):
#                 encoder_blks.append(('enc{}_{}'.format(i,c), Cell(cin, self.comp_graphs)))
#             if i < n_blk - 1:
#                 cout = cin * 2
#                 encoder_blks.append(('dpool{}'.format(i), conv2dpool(cin, cout, pool_type='max')))
#                 self.channels = cout
#         return nn.Sequential(OrderedDict(encoder_blks))
#
#     def decoder(self):
#         decoder_blks = []
#         n_blk = 4
#         for i in range(n_blk):
#             cin = self.channels
#             if i < n_blk - 1:
#                 cout = cin // 2
#                 n = n_blk - (1 + i)
#                 decoder_blks.append(('upool{}'.format(n), conv2dtransp(cin, cout)))
#                 for c in range(1):
#                     decoder_blks.append(('dec{}_{}'.format(n,c), Cell(cout, self.comp_graphs)))
#                 self.channels = cout
#             else:
#                 n = n_blk - (1 + i)
#                 for c in range(1):
#                     decoder_blks.append(('dec{}_{}'.format(n,c), Cell(self.channels, self.comp_graphs)))
#         return nn.Sequential(OrderedDict(decoder_blks))
#
#     def head(self):
#         cin = self.channels
#         return init_default(nn.Conv2d(cin, self.classes, 1), nn.init.kaiming_normal_)
#
#     def model(self):
#         stem = self.stem
#         enc0 = self.enc0_0
#         dpool0 = self.dpool0
#         enc1 = self.enc1_0
#         dpool1 = self.dpool1
#         enc2 = self.enc2_0
#         dpool2 = self.dpool2
#         enc3 = self.enc3_0
#
#         upool3 = self.upool3
#         sum3_2 = torch.sum(upool3, enc2)
#         dec3 = self.dec3_0
#         upool2 = self.upool2
#         sum2_1 = upool2 + enc1
#         dec2 = self.dec2_0
#         upool1 = self.upool1
#         sum1_0 = upool1 + enc0
#         dec1 = self.dec1_0
#         sum_0 = dec1 + stem
#         dec0 = self.dec0_0
#         return nn.Sequential(stem, enc0, dpool0)

def get_net(model_config):
    return Network(model_config)

# functions exported
__all__ = ['get_net', 'arch_config', 'Network']
