from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Concatenate, Add, Multiply, Dropout, Activation, BatchNormalization 

def Attention_block(gating_signal, skip_connection, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(gating_signal)
    #g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(skip_connection)
    #x1 = BatchNormalization()(x1)
    psi = Activation('relu')(Add()([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    #psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = Multiply()([skip_connection, psi])
    return out


def conv_block2(n_filter, n1, n2,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func



def conv_block3(n_filter, n1, n2, n3,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):

    def _func(lay):
        if batch_norm:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv3D(n_filter, (n1, n2, n3), padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func

def attunet_block(n_depth=2, n_filter_base=16, kernel_size=(3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=(2,2),
               kernel_init="glorot_uniform",
               expansion=2,
               prefix=''):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block = conv_block2 if n_dim == 2 else conv_block3
    pooling = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(int(n_filter_base * expansion ** n), *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   init=kernel_init,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(int(n_filter_base * expansion ** n_depth), *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

        layer = conv_block(int(n_filter_base * expansion ** max(0, n_depth - 1)), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           init=kernel_init,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

        for n in reversed(range(n_depth)):
            print(n)
            print(int(n_filter_base * expansion ** n))

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            gating_signal = upsampling(pool)(layer)
            skip_connection = skip_layers[n]
            attn_skip_connection = Attention_block(skip_connection, gating_signal, int(n_filter_base * expansion ** n))

            layer = Concatenate(axis=channel_axis)([gating_signal, attn_skip_connection])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(int(n_filter_base * expansion ** n), *kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(int(n_filter_base * expansion ** max(0, n - 1)), *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func

