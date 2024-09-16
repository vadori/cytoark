from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import warnings
import math
from tqdm import tqdm
import tensorflow as tf
import os
import time

from csbdeep.internals.blocks import unet_block, resnet_block
from csbdeep.utils import _raise
from csbdeep.utils.tf import keras_import

keras = keras_import()
K = keras_import('backend')
Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D', 'Concatenate', 'UpSampling2D')
Model = keras_import('models', 'Model')

from cisca.architectures._mrunet import mrunet_block
from cisca.architectures._fpn import fpn_block
from cisca.architectures._unetplus import unet_block as unet_block_v2
from cisca.architectures._attunet import attunet_block

def get_unetmodel(input_shape, n_contour_classes, n_celltype_classes, backbone, dist_regression, diag_dist, head_blocks, unet_grid, unet_n_conv_per_depth, unet_n_filter_base, 
                  unet_kernel_size, unet_activation, net_conv_after_unet, unet_last_activation, unet_n_depth, unet_pool, unet_batch_norm, unet_dropout, unet_prefix):

    # https://stackoverflow.com/questions/2521901/get-a-list-tuple-dict-of-the-arguments-passed-to-a-function
    # This is a bit hackish, however, as locals() returns all variables in the local scope
    unet_kwargs = {k[len('unet_'):]:v for (k,v) in locals().items() if k.startswith('unet_')}
    del unet_kwargs["grid"]
    
    def _head(x, filters=32):
        for _i in range(head_blocks):
            x = resnet_block(filters)(x)
        return x 
    
    if dist_regression:
        if diag_dist:
            grad_channels = 4
        else:
            grad_channels = 2

    input_img = Input(input_shape, name='input')

    # maxpool input image to unet_grid size
    pooled = np.array([1,1])
    pooled_img = input_img
    while tuple(pooled) != tuple(unet_grid):
        pool = 1 + (np.asarray(unet_grid) > pooled)
        pooled *= pool
        for _ in range(unet_n_conv_per_depth):
            pooled_img = Conv2D(unet_n_filter_base, unet_kernel_size,
                                padding='same', activation=unet_activation)(pooled_img)
        pooled_img = MaxPooling2D(pool)(pooled_img)

    if backbone=='unet':
        unet_base = unet_block(**unet_kwargs)(pooled_img)
    elif backbone=='attunet':
        unet_base = attunet_block(**unet_kwargs)(pooled_img)
    elif backbone=='unetplus':
        print(unet_kwargs)
        unet_base = unet_block_v2(n_depth=unet_kwargs['n_depth'],
                                n_filter_base=unet_kwargs['n_filter_base'],
                                kernel_size=(3,3),
                                strides = unet_kwargs['pool'],
                                block='conv_bottleneck',
                                n_blocks=2,
                                expansion=1.5,
                                multi_heads = True,
                                activation="elu",
                                batch_norm=True)(pooled_img)
        unet_base = tuple(UpSampling2D(tuple(p**i for p in unet_kwargs['pool']), interpolation='bilinear')(x) for i,x in enumerate(unet_base))
        unet_base = Concatenate()(unet_base)
        
    elif backbone=='mrunet':
        unet_base = mrunet_block(unet_kwargs['n_filter_base'])(pooled_img)
    elif backbone=='fpn':
        unet_base = fpn_block(head_filters=unet_kwargs['n_filter_base'],
                                **unet_kwargs)(pooled_img)
    else:
        _raise(NotImplementedError(backbone))

    if net_conv_after_unet > 0:
        unet = Conv2D(net_conv_after_unet, unet_kernel_size,
                        name='features', padding='same', activation=unet_activation)(unet_base)
    else:
        unet = unet_base
            
    output_contour = Conv2D(n_contour_classes, (1,1), name='contour', padding='same', activation='softmax')(_head(unet, unet_n_filter_base//2))
    output_dist = Conv2D(grad_channels, (1,1), name='dist', padding='same', activation='linear')(_head(unet, grad_channels))

    # attach extra classification head when multiclass
    if n_celltype_classes > 1:
        if net_conv_after_unet > 0:
            unet_class  = Conv2D(net_conv_after_unet,
                            unet_kernel_size,
                            name='features_class', padding='same',
                            activation=unet_activation)(unet_base)
        else:
            unet_class  = unet_base

        output_prob_class  = Conv2D(n_celltype_classes+1, (1,1), name='prob_class', padding='same', activation='softmax')(_head(unet_class,unet_n_filter_base//2))
        
        return Model([input_img], [output_contour,output_dist,output_prob_class])
    else:
        return Model([input_img], [output_contour,output_dist])

