"""This module 'convolution_operations' provides an interface for convolutional neural network (CNN)
operations that have been implemented
"""
from collections import namedtuple

CONV_CODES = [('conv1x1', 'A'), ('conv3x3', 'B'), ('dwconv3x3', 'C'), ('conv1x3', 'D'), ('conv3x1', 'F'),
              ('maxpool3x3', 'G')]

CONV_OPS = namedtuple('CONV_OPS', ['conv1x1', 'conv3x3','dwconv3x3','conv1x3','conv3x1', 'maxpool3x3'])

CONV_OPS.__new__.__defaults__ = (None,) * len(CONV_CODES)

def get_operation():
    return CONV_OPS._fields

def get_symbol():
    return CONV_OPS(*CONV_CODES)

operations = CONV_OPS._fields

__all__ = ['operations', 'get_operation', 'get_symbol']
