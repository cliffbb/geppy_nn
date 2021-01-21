"""
"""
from fastai.vision.all import *
from collections import namedtuple
from gepnet.utils import *

arch_config = namedtuple('arch_config', ['comp_graphs', 'channels', 'repeat_list', 'classes'])
arch_config.__new__.__defaults__ = (None,) * len(arch_config._fields)


class CompGraph(nn.Module):
    """Class that is used to build a layer"""
    def __init__(self, cin, comp_graph):
        super(CompGraph, self).__init__()
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
            # elif get_op_head(op) == 'conv3x1x3':
            #     self.add_module(op, conv2d_asy(cin))
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
    "Class that is used to build a cell"
    def __init__(self, cin, comp_graphs):
        super(Cell, self).__init__()
        self.n_branch = len(comp_graphs)
        # self.relu = nn.ReLU(True)
        for i in range(self.n_branch):
            setattr(self, 'gene_%d' % i, CompGraph(cin, comp_graphs[i]))
        # for i in range(self.n_branch):
        #     layers.append(CompGraph(cin, comp_graph[i]))
        # # self.proj = conv2d(cin*self.n_branch, cin, ksize=1)
        # self.layers = nn.Sequential(*torch.stack(layers))

    def forward(self, x):
        cell = [] #[None] * self.n_branch
        for i in range(self.n_branch):
            cell.append(getattr(self, 'gene_%d' % i)(x))
        # x = self.proj(concat(self.layers(x)))
        # x = self.cell(x)
        # print('concat:', concat(*cell).shape, x.shape)
        return cat(*cell) #self.relu(cell + x)



# hidden_dim = in_planes * expand_ratio
# reduced_dim = max(1, int(in_planes / reduction_ratio))
# class SqueezeExcitation(nn.Module):
#
#     def __init__(self, in_planes, reduced_dim):
#         super(SqueezeExcitation, self).__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_planes, reduced_dim, 1),
#             Swish(),
#             nn.Conv2d(reduced_dim, in_planes, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         return x * self.se(x)
# # Squeeze and Excitation
# if self.has_se:
#     x_squeezed = F.adaptive_avg_pool2d(x, 1)
#     x_squeezed = self._se_reduce(x_squeezed)
#     x_squeezed = self._swish(x_squeezed)
#     x_squeezed = self._se_expand(x_squeezed)
#     x = torch.sigmoid(x_squeezed) * x
#
# # Squeeze and Excitation layer, if desired
# if self.has_se:
#     Conv2d = get_same_padding_conv2d(image_size=(1, 1))
#     num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
#     self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
#     self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
#
# class SqEx(nn.Module):
#     def __init__(self, n_features, reduction=16):
#         super(SqEx, self).__init__()
#
#         if n_features % reduction != 0:
#             raise ValueError('n_features must be divisible by reduction (default = 16)')
#
#         self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
#         self.nonlin1 = nn.ReLU(inplace=True)
#         self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
#         self.nonlin2 = nn.Sigmoid()
#
#     def forward(self, x):
#         y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
#         y = y.permute(0, 2, 3, 1)
#         y = self.nonlin1(self.linear1(y))
#         y = self.nonlin2(self.linear2(y))
#         y = y.permute(0, 3, 1, 2)
#         y = x * y
#         return y


# # Cell
# class ProdLayer(Module):
#     "Merge a shortcut with the result of the module by multiplying them."
#     def forward(self, x): return x * x.orig
#
# # Cell
# def SEModule(ch, reduction, act_cls=defaults.activation):
#     nf = math.ceil(ch//reduction/8)*8
#     return SequentialEx(nn.AdaptiveAvgPool2d(1),
#                         ConvLayer(ch, nf, ks=1, norm_type=None, act_cls=act_cls),
#                         ConvLayer(nf, ch, ks=1, norm_type=None, act_cls=nn.Sigmoid),
#                         ProdLayer())


class Network(nn.Module):
    """Class that is used to build the entire architecture"""
    def __init__(self, model_config):
        super(Network, self).__init__()
        self.comp_graphs = model_config.comp_graphs
        self.channels = model_config.channels
        self.repeat_list = model_config.repeat_list
        self.classes = model_config.classes

        self.stem = self.stem_()
        self.blocks = self.blocks_()
        self.head = self.head_()

    def stem_(self):
        stem = stem_blk(cin=3, cout=self.channels, ksize=3, pool1='max', double_stack=False)
        # self.channels *= 2
        return stem

    def blocks_(self):
        blocks = []
        n_blocks = len(self.repeat_list)
        for blk in range(n_blocks):
            cin = self.channels
            n_cells = self.repeat_list[blk]
            if n_cells == 1:
                blocks.append(Cell(cin, self.comp_graphs))
                blocks.append(SEModule(cin * len(self.comp_graphs), 16))
                if blk < n_blocks - 1:
                    blocks.append(conv2dpool(cin * len(self.comp_graphs), cin * 2, pool_type='max'))
                    self.channels *= 2
                else:
                    blocks.append(conv2d(cin * len(self.comp_graphs), cin * len(self.comp_graphs), ksize=1))
                    self.channels *= len(self.comp_graphs)
            # cin = self.channels
            # # cout = cin
            # n_cells = self.repeat_list[blk]
            # if n_cells == 1:
            #     # cin = self.channels
            #     blocks.append(Cell(cin, self.comp_graphs))
            #     # blocks.append(SEModule(cin * len(self.comp_graphs), 8))
            #     blocks.append(conv2d(cin * len(self.comp_graphs), cin, ksize=1))
            #     cout = cin * 2
            #     blocks.append(conv2dpool(cin, cout, pool_type='max'))
            #     self.channels = cout
            #     # print(blk, cin, cout)
            else:
                for cell in range(n_cells):
                    blocks.append(Cell(cin, self.comp_graphs))  ##
                    # blocks.append(SEModule(cin * len(self.comp_graphs), 4))
                    blocks.append(conv2d(cin * len(self.comp_graphs), cin, ksize=1)) ##)
                    # print(blk, cin)
                cout = cin * 2
                blocks.append(conv2dpool(cin, cout, pool_type='max'))
                self.channels = cout
        return nn.Sequential(*blocks)

    def head_(self):
        cin = self.channels
        return nn.Sequential(nn.AdaptiveAvgPool2d(1),
                             nn.Dropout2d(0.5),
                             Flatten(),
                             nn.Linear(cin, self.classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def get_net(model_config):
    return Network(model_config)

# functions exported
__all__ = ['get_net', 'arch_config', 'Network']
