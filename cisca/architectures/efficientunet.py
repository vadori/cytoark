import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import models
from .efficientnet import *
from .utils import conv_kernel_initializer
import tensorflow.keras.backend as K



__all__ = ['get_efficient_unet_b0', 'get_efficient_unet_b1', 'get_efficient_unet_b2', 'get_efficient_unet_b3',
           'get_efficient_unet_b4', 'get_efficient_unet_b5', 'get_efficient_unet_b6', 'get_efficient_unet_b7',
           'get_blocknr_of_skip_candidates']


def get_blocknr_of_skip_candidates(encoder, block_lookup, verbose=False):
    """
    Get block numbers of the blocks which will be used for concatenation in the Unet.
    :param encoder: the encoder
    :param verbose: if set to True, the shape information of all blocks will be printed in the console
    :return: a list of block numbers
    """
    shapes = []
    candidates = []
    mbblock_nr = 0
    while True:
            try:
                if block_lookup[mbblock_nr] > 0:
                    mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
                    shape = int(mbblock.shape[1]), int(mbblock.shape[2])
                    if shape not in shapes:
                        shapes.append(shape)
                        candidates.append(mbblock_nr)
                    # else:
                    #     candidates.pop(-1)
                    #     candidates.append(mbblock_nr)
                    if verbose:
                        print('blocks_{}_{}__output_shape: {}'.format(block_lookup[mbblock_nr],mbblock_nr, shape))
                mbblock_nr += 1
            except Exception as e: 
                print(e)
            # except ValueError:
                break
    return candidates


def DoubleConv(filters, kernel_size, initializer='glorot_uniform'):

    def layer(x):

        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    return layer

def SingleConv(filters, kernel_size, initializer='glorot_uniform'):

    def layer(x):

        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    return layer

def DoubleConvReg(filters, kernel_size, initializer='glorot_uniform'):

    def layer(x):

        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Conv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer=initializer)(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        
        return x

    return layer


def UpSampling2D_block(filters, kernel_size=(3, 3), upsample_rate=(2, 2), interpolation='bilinear',
                       initializer='glorot_uniform', skip=None, attention=None, final_conv=True, atten_activation="relu", name=None):
    
    def layer(input_tensor):

        x = UpSampling2D(size=upsample_rate, interpolation=interpolation)(input_tensor)

        if skip is not None:
            
            if attention is not None:

                skipatt = attention_gate(X=skip, g=x, channel=filters//2, activation=atten_activation, 
                                    attention=attention, name = name + "_att")
        
                # Tensor concatenation
                x = Concatenate()([x, skipatt])
                #H = concatenate([X, X_left], axis=-1, name='{}_concat'.format(name))
                
                # stacked linear convolutional layers after concatenation
                #x = CONV_stack(x, filters, kernel_size, stack_num=stack_num, activation=activation, 
                #            batch_norm=batch_norm, name='{}_conv_after_concat'.format(name))

            else:
                x = Concatenate()([x, skip])

        #if skip is not None:
        #    x = Concatenate()([x, skip])
        if final_conv:
            x = DoubleConv(filters, kernel_size, initializer=initializer)(x)

        return x
    return layer



def Conv2DTranspose_block(filters, kernel_size=(3, 3), transpose_kernel_size=(2, 2), upsample_rate=(2, 2),
                          initializer='glorot_uniform', skip=None):
    def layer(input_tensor):

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same')(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        x = DoubleConv(filters, kernel_size, initializer=initializer)(x)

        return x

    return layer


def Conv2DTranspose_block_last(filters, kernel_size=(3, 3), transpose_kernel_size=(2, 2), upsample_rate=(2, 2),
                          initializer='glorot_uniform', skip=None):
    def layer(input_tensor):

        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate, padding='same')(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        return x

    return layer

def UpSampling2D_block_last(filters, kernel_size=(3, 3), upsample_rate=(2, 2), interpolation='bilinear',
                       initializer='glorot_uniform', skip=None):
    def layer(input_tensor):

        x = UpSampling2D(size=upsample_rate, interpolation=interpolation)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

        return x

    return layer

# noinspection PyTypeChecker
def _get_CISCA_unet(encoder, block_lookup, n_contour_classes=3, block_type='upsampling', concat_input=True, grad_channels = 4):
    
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)
    print(skip_candidates)

    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        print(mbblock)
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1=blocks.pop()
    skip2=blocks.pop()
    skip3=blocks.pop()
    skip4=blocks.pop()
    
    o = UpBlock(512, initializer=conv_kernel_initializer, skip = skip1, attention="add", name = "upblk11")(ostart)
    o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention="add", name = "upblk12")(ostart)

    otmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention="add",final_conv = False, name = "upblk21")(o)
    otmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention="add",final_conv = False, name = "upblk22")(o2)
    o2tmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)

    skip = attention_gate(X=o2tmp, g=otmp, channel=128, activation="relu", 
                                    attention="add", name = "ii1")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=128, activation="relu", 
                                    attention="add", name = "ii2")
    # Tensor concatenation
    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip2])
    o = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention="add",final_conv = False, name = "upblk31")(o)
    otmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention="add",final_conv = False, name = "upblk32")(o2)
    o2tmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=64, activation="relu", 
                                    attention="add", name = "ii3")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=64, activation="relu", 
                                    attention="add", name = 'ii4')
    # Tensor concatenation
    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip2])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    o = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4,attention="add",name = "upblk41")(o)
    o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4,attention="add",name = "upblk42")(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    if n_contour_classes == 3:
        out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    else:
        out_center = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation ='softmax', name ='out_center')(out_center)
    #print(out_center.shape)

    # # second head added 
    # o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1)(ostart)
    # o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2)(o2)
    # o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3)(o2)
    # o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4)(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center2')(out_center2)
    #print(out_center.shape)

    # remove out_center2 to go back to original 
    model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model

def _get_CISCA_unet2(encoder, block_lookup, n_contour_classes=3, block_type='upsampling', concat_input=True, grad_channels = 4, attention = "add", activationlast = 'softmax'):
    
    print("Creating unet version 2")
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)
    print(skip_candidates)

    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        print("mbblock:",mbblock)
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1=blocks.pop()
    skip2=blocks.pop()
    skip3=blocks.pop()
    skip4=blocks.pop()
    
    #o = UpBlock(512, initializer=conv_kernel_initializer, skip = skip1, attention="add", name = "upblk11")(ostart)
    #o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention="add", name = "upblk12")(ostart)

    print(ostart.shape)
    print(skip1.shape)

    otmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention=attention,name = "upblk11")(ostart)
    #otmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention=attention,name = "upblk12")(ostart)
    #o2tmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)

    skip = attention_gate(X=o2tmp, g=otmp, channel=256, activation="relu", 
                                    attention="add", name = "ii0")
    skip1 = attention_gate(X=otmp, g=o2tmp, channel=256, activation="relu", 
                                    attention="add", name = "ii1")
    # Tensor concatenation
    print("otmp", otmp.shape)
    print(skip.shape)

    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip1])
    o = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    print(o.shape)
    print(skip1.shape)

    otmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention=attention,name = "upblk21")(o)
    #otmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention=attention,name = "upblk22")(o2)
    #o2tmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)

    skip = attention_gate(X=o2tmp, g=otmp, channel=128, activation="relu", 
                                    attention="add", name = "ii2")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=128, activation="relu", 
                                    attention="add", name = "ii3")
    # Tensor concatenation
    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip2])
    o = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention=attention,name = "upblk31")(o)
    #otmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention=attention,name = "upblk32")(o2)
    #o2tmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=64, activation="relu", 
                                    attention="add", name = "ii4")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=64, activation="relu", 
                                    attention="add", name = 'ii5')
    # Tensor concatenation
    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip2])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4, attention=attention,name = "upblk41")(o)
    #otmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4, attention=attention,name = "upbl42")(o2)
    #o2tmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=32, activation="relu", 
                                    attention="add", name = "ii6")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=32, activation="relu", 
                                    attention="add", name = 'ii7')
    # Tensor concatenation
    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip2])
    o = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #if n_contour_classes == 3:
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #else:
    #    out_center = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)

    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation = activationlast, name ='out_center')(out_center)
    #print(out_center.shape)

    # # second head added 
    # o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1)(ostart)
    # o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2)(o2)
    # o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3)(o2)
    # o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4)(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center2')(out_center2)
    #print(out_center.shape)

    # remove out_center2 to go back to original 
    model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model

def _get_CISCA_unet3(encoder, block_lookup, n_contour_classes=3, n_celltype_classes=1,block_type='upsampling', concat_input=True, grad_channels = 4, attention = "add", activationlast = 'softmax'):
    
    print("Creating unet version 3")
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)
    print(skip_candidates)

    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        print("mbblock:",mbblock)
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1enc=blocks.pop()
    skip2enc=blocks.pop()
    skip3enc=blocks.pop()
    skip4enc=blocks.pop()
    
    #o = UpBlock(512, initializer=conv_kernel_initializer, skip = skip1, attention="add", name = "upblk11")(ostart)
    #o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention="add", name = "upblk12")(ostart)

    print(ostart.shape)
    print(skip1enc.shape)

    otmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1enc, attention=attention,name = "upblk11")(ostart)
    #otmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1enc, attention=attention,name = "upblk12")(ostart)
    #o2tmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
    
    skip = attention_gate(X=o2tmp, g=otmp, channel=256, activation="relu", 
                                    attention="add", name = "at112")
    skip1 = attention_gate(X=otmp, g=o2tmp, channel=256, activation="relu", 
                                    attention="add", name = "at121")
    if n_celltype_classes > 1:
        o3tmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1enc, attention=attention,name = "upblk13")(ostart)
        skip3 = attention_gate(X=otmp, g=o3tmp, channel=256, activation="relu", 
                                    attention="add", name = "at131")
    # Tensor concatenation
    print("otmp", otmp.shape)
    print(skip.shape)

    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip1])
    o = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
    if n_celltype_classes > 1:
        o3 = Add()([o3tmp, skip3])
        o3 = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o3)

    print(o.shape)
    print(skip1.shape)

    otmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2enc, attention=attention,name = "upblk21")(o)
    #otmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2enc, attention=attention,name = "upblk22")(o2)
    #o2tmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)

    skip = attention_gate(X=o2tmp, g=otmp, channel=128, activation="relu", 
                                    attention="add", name = "at212")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=128, activation="relu", 
                                    attention="add", name = "at221")
    
    if n_celltype_classes > 1:
        o3tmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2enc, attention=attention,name = "upblk23")(o3)
        skip3 = attention_gate(X=otmp, g=o3tmp, channel=128, activation="relu", 
                                    attention="add", name = "at231")

    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
    if n_celltype_classes > 1:
        o3 = Add()([o3tmp, skip3])
        o3 = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o3)


    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk31")(o)
    #otmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk32")(o2)
    #o2tmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=64, activation="relu", 
                                    attention="add", name = "at312")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=64, activation="relu", 
                                    attention="add", name = 'at321')
    
    if n_celltype_classes > 1:
        o3tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk33")(o3)
        skip3 = attention_gate(X=otmp, g=o3tmp, channel=64, activation="relu", 
                                    attention="add", name = "at331")


    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
    if n_celltype_classes > 1:
        o3 = Add()([o3tmp, skip3])
        o3 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o3)


    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upblk41")(o)
    #otmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upbl42")(o2)
    #o2tmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=32, activation="relu", 
                                    attention="add", name = "at412")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=32, activation="relu", 
                                    attention="add", name = 'at421')
    if n_celltype_classes > 1:
        o3tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upblk43")(o3)
        skip3 = attention_gate(X=otmp, g=o3tmp, channel=32, activation="relu", 
                                    attention="add", name = "at431")
    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    if n_celltype_classes > 1:
        o3 = Add()([o3tmp, skip3])
        o3 = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o3)

    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #if n_contour_classes == 3:
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #else:
    #    out_center = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)

    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation = activationlast, name ='out_center')(out_center)
    #print(out_center.shape)

    # # second head added 
    # o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1)(ostart)
    # o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2)(o2)
    # o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3)(o2)
    # o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4)(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center2')(out_center2)
    #print(out_center.shape)

    if n_celltype_classes > 1:
        if concat_input:
            # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
            # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
            o3 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o3)
        else:
            o3 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o3)

        out_center3 = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o3)
    
        out_center3 = Conv2D(n_celltype_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation = activationlast, name ='out_center3')(out_center3)
        #print(out_center.shape)

        # remove out_center2 to go back to original 
        model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2, out_center3])
    
    else:
        model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model

def _get_CISCA_unet3_reduced(encoder, block_lookup, n_contour_classes=3, n_celltype_classes=1,block_type='upsampling', concat_input=True, grad_channels = 4, attention = "add", activationlast = 'softmax'):
    
    print("Creating unet version 3 reduced")
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)
    print(skip_candidates)

    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        print("mbblock:",mbblock)
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1enc=blocks.pop()
    skip2enc=blocks.pop()
    skip3enc=blocks.pop()
    skip4enc=blocks.pop()
    
    #o = UpBlock(512, initializer=conv_kernel_initializer, skip = skip1, attention="add", name = "upblk11")(ostart)
    #o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention="add", name = "upblk12")(ostart)

    print(ostart.shape)
    print(skip1enc.shape)

    otmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1enc, attention=attention,name = "upblk11")(ostart)
    #otmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1enc, attention=attention,name = "upblk12")(ostart)
    #o2tmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
    
    skip = attention_gate(X=o2tmp, g=otmp, channel=256, activation="relu", 
                                    attention="add", name = "at112")
    skip1 = attention_gate(X=otmp, g=o2tmp, channel=256, activation="relu", 
                                    attention="add", name = "at121")

    # Tensor concatenation
    print("otmp", otmp.shape)
    print(skip.shape)

    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip1])
    o = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    print(o.shape)
    print(skip1.shape)

    otmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2enc, attention=attention,name = "upblk21")(o)
    #otmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2enc, attention=attention,name = "upblk22")(o2)
    #o2tmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)

    skip = attention_gate(X=o2tmp, g=otmp, channel=128, activation="relu", 
                                    attention="add", name = "at212")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=128, activation="relu", 
                                    attention="add", name = "at221")

    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
   
    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk31")(o)
    #otmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk32")(o2)
    #o2tmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=64, activation="relu", 
                                    attention="add", name = "at312")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=64, activation="relu", 
                                    attention="add", name = 'at321')
    
    # branching!
    if n_celltype_classes > 1:
        o3 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk33")(o)
        # skip3 = attention_gate(X=otmp, g=o3tmp, channel=64, activation="relu", 
        #                             attention="add", name = "at331")

    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
    # if n_celltype_classes > 1:
    #     o3 = Add()([o3tmp, skip3])
    #     o3 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o3)


    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upblk41")(o)
    #otmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upbl42")(o2)
    #o2tmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=32, activation="relu", 
                                    attention="add", name = "at412")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=32, activation="relu", 
                                    attention="add", name = 'at421')
    if n_celltype_classes > 1:
        o3tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upblk43")(o3)
        skip3 = attention_gate(X=otmp, g=o3tmp, channel=32, activation="relu", 
                                    attention="add", name = "at431")
    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    if n_celltype_classes > 1:
        o3 = Add()([o3tmp, skip3])
        o3 = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o3)

    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #if n_contour_classes == 3:
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #else:
    #    out_center = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)

    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_centerpre')(out_center)
    #print(out_center.shape)

    # # second head added 
    # o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1)(ostart)
    # o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2)(o2)
    # o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3)(o2)
    # o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4)(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center2')(out_center2)
    #print(out_center.shape)

    if n_celltype_classes > 1:
        if concat_input:
            # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
            # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
            o3 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o3)
        else:
            o3 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o3)

        out_center3 = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o3)
    
        out_center3 = Conv2D(n_celltype_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center3pre')(out_center3)
        out_center3 = tf.nn.softmax(Concatenate()([tf.reduce_sum(tf.keras.layers.Lambda(lambda x : x[..., 2:])(out_center),axis = -1, keepdims=True), out_center3]),name ='out_center3')
        #out_center3 = tf.nn.softmax(Concatenate()([tf.reduce_sum(out_center[..., 2:],axis = -1, keepdims=True), out_center3]))
        out_center = tf.nn.softmax(out_center,name ='out_center')
   
        #print(out_center.shape)

        # remove out_center2 to go back to original 
        model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2, out_center3])
    
    else:
        model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model


from keras.layers import Conv2D, Lambda, Dense, Multiply, Add
import keras.backend as K

class scSE:
    
    def __init__(self, layer):
        self.layer = layer
        self.layer_shape = K.int_shape(layer)
        
        
    def sSE(self):
        channel_squeeze = Conv2D(1, (1,1), kernel_initializer="he_normal", activation='sigmoid')(self.layer)
        channel_squeeze = Multiply()([self.layer, channel_squeeze])
        return channel_squeeze
    
    
    def cSE(self):

        spatial_squeeze = Lambda(lambda x :K.mean(x, axis = [1,2]))(self.layer)
        spatial_squeeze = Dense(self.layer_shape[-1]//2, activation='relu')(spatial_squeeze)
        spatial_squeeze = Dense(self.layer_shape[-1], activation='sigmoid')(spatial_squeeze)
        spatial_squeeze = Multiply()([spatial_squeeze, self.layer])
        return spatial_squeeze
        
    
    def scSE(self):
        channel_squeeze = self.cSE()
        spatial_squeeze = self.sSE()
        scse = Add()([channel_squeeze, spatial_squeeze])
        
        return scse
        

    
def _get_CISCA_unet3_reduced_squeezeexcite(encoder, block_lookup, n_contour_classes=3, n_celltype_classes=1,block_type='upsampling', concat_input=True, grad_channels = 4, attention = "add", activationlast = 'softmax'):
    
    print("Creating unet version 3 reduced with squeeze")
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)
    print(skip_candidates)

    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        print("mbblock:",mbblock)
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1enc=scSE(blocks.pop()).scSE()
    skip2enc=scSE(blocks.pop()).scSE()
    skip3enc=scSE(blocks.pop()).scSE()
    skip4enc=scSE(blocks.pop()).scSE()
    
    #o = UpBlock(512, initializer=conv_kernel_initializer, skip = skip1, attention="add", name = "upblk11")(ostart)
    #o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention="add", name = "upblk12")(ostart)

    print(ostart.shape)
    print(skip1enc.shape)

    otmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1enc, attention=attention,name = "upblk11")(ostart)
    #otmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1enc, attention=attention,name = "upblk12")(ostart)
    #o2tmp = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
    
    skip = attention_gate(X=o2tmp, g=otmp, channel=256, activation="relu", 
                                    attention="add", name = "at112")
    skip1 = attention_gate(X=otmp, g=o2tmp, channel=256, activation="relu", 
                                    attention="add", name = "at121")

    # Tensor concatenation
    print("otmp", otmp.shape)
    print(skip.shape)

    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip1])
    o = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(512, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    print(o.shape)
    print(skip1.shape)

    otmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2enc, attention=attention,name = "upblk21")(o)
    #otmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2enc, attention=attention,name = "upblk22")(o2)
    #o2tmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)

    skip = attention_gate(X=o2tmp, g=otmp, channel=128, activation="relu", 
                                    attention="add", name = "at212")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=128, activation="relu", 
                                    attention="add", name = "at221")

    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
   
    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk31")(o)
    #otmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk32")(o2)
    #o2tmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=64, activation="relu", 
                                    attention="add", name = "at312")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=64, activation="relu", 
                                    attention="add", name = 'at321')
    
    # branching!
    if n_celltype_classes > 1:
        o3 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3enc, attention=attention,name = "upblk33")(o)
        # skip3 = attention_gate(X=otmp, g=o3tmp, channel=64, activation="relu", 
        #                             attention="add", name = "at331")

    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
    # if n_celltype_classes > 1:
    #     o3 = Add()([o3tmp, skip3])
    #     o3 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o3)


    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upblk41")(o)
    #otmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upbl42")(o2)
    #o2tmp = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=32, activation="relu", 
                                    attention="add", name = "at412")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=32, activation="relu", 
                                    attention="add", name = 'at421')
    if n_celltype_classes > 1:
        o3tmp = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4enc, attention=attention,name = "upblk43")(o3)
        skip3 = attention_gate(X=otmp, g=o3tmp, channel=32, activation="relu", 
                                    attention="add", name = "at431")
    # Tensor concatenation
    o = Add()([otmp, skip])
    o2 = Add()([o2tmp, skip2])
    o = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    if n_celltype_classes > 1:
        o3 = Add()([o3tmp, skip3])
        o3 = DoubleConv(64, kernel_size=(3, 3), initializer='glorot_uniform')(o3)

    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #if n_contour_classes == 3:
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #else:
    #    out_center = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)

    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_centerpre')(out_center)
    #print(out_center.shape)

    # # second head added 
    # o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1)(ostart)
    # o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2)(o2)
    # o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3)(o2)
    # o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4)(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_dist_regress')(out_center2)
    #out_center2 = Subtract(name ='out_dist_regress')([4.33*Activation('sigmoid')(out_center2),2.165])
    #print(out_center.shape)out_dist_regress

    if n_celltype_classes > 1:
        if concat_input:
            # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
            # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
            o3 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o3)
        else:
            o3 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o3)

        out_center3 = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o3)
    
        out_center3 = Conv2D(n_celltype_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center3pre')(out_center3)
        out_center3 = Softmax(name ='out_cell_type_class')(Concatenate()([tf.reduce_sum(tf.keras.layers.Lambda(lambda x : x[..., 2:])(out_center),axis = -1, keepdims=True), out_center3]))
        #out_center3 = tf.nn.softmax(Concatenate()([tf.reduce_sum(out_center[..., 2:],axis = -1, keepdims=True), out_center3]))
        out_center = Softmax(name ='out_cell_inst_seg')(out_center)
   
        #print(out_center.shape)

        # remove out_center2 to go back to original 
        model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2, out_center3])
    
    else:
        out_center = Softmax(name ='out_cell_inst_seg')(out_center)
        model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model

def _get_efficient_unet_noattdecoder(encoder, block_lookup, n_contour_classes=3, block_type='upsampling', concat_input=True, grad_channels = 4):
    
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)
    print(skip_candidates)

    for mbblock_nr in skip_candidates:
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        print(mbblock)
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1=blocks.pop()
    skip2=blocks.pop()
    skip3=blocks.pop()
    skip4=blocks.pop()
    
    o = UpBlock(512, initializer=conv_kernel_initializer, skip = skip1, attention="add", name = "upblk11")(ostart)
    o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention="add", name = "upblk12")(ostart)

    otmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention="add",final_conv = False, name = "upblk21")(o)
    otmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention="add",final_conv = False, name = "upblk22")(o2)
    o2tmp = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)

    skip = attention_gate(X=o2tmp, g=otmp, channel=128, activation="relu", 
                                    attention="add", name = "ii1")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=128, activation="relu", 
                                    attention="add", name = "ii2")
    # Tensor concatenation
    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip2])
    o = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(256, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention="add",final_conv = False, name = "upblk31")(o)
    otmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(otmp)
    o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention="add",final_conv = False, name = "upblk32")(o2)
    o2tmp = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2tmp)
     
    # otmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk31")(o)
    # o2tmp = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3,attention="add",final_conv = False,name = "upblk32")(o2)

    skip = attention_gate(X=o2tmp, g=otmp, channel=64, activation="relu", 
                                    attention="add", name = "ii3")
    skip2 = attention_gate(X=otmp, g=o2tmp, channel=64, activation="relu", 
                                    attention="add", name = 'ii4')
    # Tensor concatenation
    o = Concatenate()([otmp, skip])
    o2 = Concatenate()([o2tmp, skip2])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    # experiment 1 (b3weights_sumweightsfromdecoderspath)
    # o =  tf.math.add(otmp, skip)
    # o2 = tf.math.add(o2tmp, skip2)

    o = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4,attention="add",name = "upblk41")(o)
    o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4,attention="add",name = "upblk42")(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation ='softmax', name ='out_center')(out_center)
    #print(out_center.shape)

    # # second head added 
    # o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1)(ostart)
    # o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2)(o2)
    # o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3)(o2)
    # o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4)(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center2')(out_center2)
    #print(out_center.shape)

    # remove out_center2 to go back to original 
    model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model

def _get_efficient_unet_noatt(encoder, block_lookup, n_contour_classes=3, block_type='upsampling', concat_input=True, grad_channels = 4, attention = None):
    
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)

    for mbblock_nr in skip_candidates:
        # mbblock = encoder.get_layer('blocks_{}_output_batch_norm'.format(mbblock_nr)).output
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1=blocks.pop()
    skip2=blocks.pop()
    skip3=blocks.pop()
    skip4=blocks.pop()

    o = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention=attention, name = "upblk11")(ostart)
    o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention=attention, name = "upblk12")(ostart)
    # Tensor concatenation (no attention)
    #o = Concatenate()([o, o2])
    #o2 = Concatenate()([o2, o])
    #o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    #o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)

    o = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention=attention,name = "upblk21")(o)
    o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention=attention,name = "upblk22")(o2)
    # Tensor concatenation (no attention)
    o = Concatenate()([o, o2])
    o2 = Concatenate()([o2, o])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)


    o = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention=attention,name = "upblk31")(o)
    o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention=attention,name = "upblk32")(o2)
    # Tensor concatenation (no attention)
    o = Concatenate()([o, o2])
    o2 = Concatenate()([o2, o])
    o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)


    o = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4, attention=attention,name = "upblk41")(o)
    o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4, attention=attention,name = "upblk42")(o2)
    # Tensor concatenation (no attention)
    #o = Concatenate()([o, o2])
    #o2 = Concatenate()([o2, o])
    #o = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o)
    #o2 = DoubleConv(128, kernel_size=(3, 3), initializer='glorot_uniform')(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
  
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation ='softmax', name ='out_center')(out_center)

    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center2')(out_center2)
    #print(out_center.shape)

    # remove out_center2 to go back to original 
    model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model

def _get_efficient_unet_noskipdecoder(encoder, block_lookup, n_contour_classes=3, block_type='upsampling', concat_input=True, grad_channels = 4, attention = "add"):
    
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)

    for mbblock_nr in skip_candidates:
        # mbblock = encoder.get_layer('blocks_{}_output_batch_norm'.format(mbblock_nr)).output
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1=blocks.pop()
    skip2=blocks.pop()
    skip3=blocks.pop()
    skip4=blocks.pop()

    o = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention=attention, name = "upblk11")(ostart)
    o = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention=attention,name = "upblk21")(o)
    o = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention=attention,name = "upblk31")(o)
    o = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4, attention=attention,name = "upblk41")(o)

    o2 = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention=attention, name = "upblk12")(ostart)
    o2 = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention=attention,name = "upblk22")(o2)
    o2 = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention=attention,name = "upblk32")(o2)
    o2 = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4, attention=attention,name = "upblk42")(o2)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
  
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation ='softmax', name ='out_center')(out_center)

    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o2)
    else:
        o2 = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o2)

    out_center2 = DoubleConvReg(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o2)
    #print(out_center.shape)
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center2 = Conv2D(grad_channels, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, name ='out_center2')(out_center2)
    #print(out_center.shape)

    # remove out_center2 to go back to original 
    model = models.Model(inputs=encoder.input, outputs=[out_center, out_center2])

    return model

def _get_efficient_unet_nograd(encoder,block_lookup, n_contour_classes=3, block_type='upsampling', concat_input=True, attention = "add"):
    
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)

    for mbblock_nr in skip_candidates:
        # mbblock = encoder.get_layer('blocks_{}_output_batch_norm'.format(mbblock_nr)).output
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    ostart=blocks.pop()
    skip1=blocks.pop()
    skip2=blocks.pop()
    skip3=blocks.pop()
    skip4=blocks.pop()

    o = UpBlock(512, initializer=conv_kernel_initializer, skip=skip1, attention=attention, name = "upblk11")(ostart)
    o = UpBlock(256, initializer=conv_kernel_initializer, skip=skip2, attention=attention,name = "upblk21")(o)
    o = UpBlock(128, initializer=conv_kernel_initializer, skip=skip3, attention=attention,name = "upblk31")(o)
    o = UpBlock(64, initializer=conv_kernel_initializer, skip=skip4, attention=attention,name = "upblk41")(o)

    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = UpSampling2D_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
  
    # out = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out')(out)
    out_center = Conv2D(n_contour_classes, (1, 1), padding = 'same', kernel_initializer = conv_kernel_initializer, activation ='softmax', name ='out_center')(out_center)

    model = models.Model(inputs=encoder.input, outputs=[out_center])

    return model

def _get_efficient_unet_framework(encoder, block_lookup, n_contour_classes=3, block_type='upsampling', concat_input=True):
    MBConvBlocks = []

    skip_candidates = get_blocknr_of_skip_candidates(encoder,block_lookup)

    for mbblock_nr in skip_candidates:
        # mbblock = encoder.get_layer('blocks_{}_output_batch_norm'.format(mbblock_nr)).output
        mbblock = encoder.get_layer('blocks_{}_{}_output_batch_norm'.format(block_lookup[mbblock_nr],mbblock_nr)).output
        MBConvBlocks.append(mbblock)

    # delete the last block since it won't be used in the process of concatenation
    MBConvBlocks.pop()

    input_ = encoder.input
    head = encoder.get_layer('head_swish').output
    blocks = [input_] + MBConvBlocks + [head]

    if block_type == 'upsampling':
        UpBlock = UpSampling2D_block
    else:
        UpBlock = Conv2DTranspose_block

    o = blocks.pop()
    o = UpBlock(512, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
    o = UpBlock(256, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
    o = UpBlock(128, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
    o = UpBlock(64, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
    
    if concat_input:
        # out = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        # out_center = UpBlock(32, initializer=conv_kernel_initializer, skip=blocks.pop())(o)
        o = Conv2DTranspose_block_last(32, initializer=conv_kernel_initializer, skip=blocks[0])(o)
    else:
        o = Conv2DTranspose_block_last(32, initializer=conv_kernel_initializer, skip=None)(o)

    # out = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)
    out_center = DoubleConv(32, kernel_size=(3, 3), initializer=conv_kernel_initializer)(o)

    out_center = Conv2D(n_contour_classes, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='softmax', name='out_center')(out_center)
    #out_center = Conv2D(1, (1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='linear', name='out_center')(out_center)

    model = models.Model(inputs=encoder.input, outputs=out_center)

    return model


def get_efficient_unet_b0(input_shape, n_contour_classes=3, pretrained=False, block_type='upsampling', concat_input=True, trainable=False, framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True, dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B0 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B0 model
    """
    # encoder, block_lookup = get_efficientnet_b0_encoder(input_shape, pretrained=pretrained)
    # model = _get_efficient_unet(encoder, n_contour_classes, block_type=block_type, concat_input=concat_input, block_lookup=block_lookup)
    # return model

    encoder, block_lookup = get_efficientnet_b0_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model


def get_efficient_unet_b1(input_shape, n_contour_classes=3, pretrained=False, block_type='upsampling', concat_input=True, trainable=False, framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True, dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B1 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B1 model
    """
    # encoder, block_lookup = get_efficientnet_b1_encoder(input_shape, pretrained=pretrained)
    # model = _get_efficient_unet(encoder, n_contour_classes, block_type=block_type, concat_input=concat_input, block_lookup=block_lookup)
    # return model

    encoder, block_lookup = get_efficientnet_b1_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model


def get_efficient_unet_b2(input_shape, n_contour_classes=3, n_celltype_classes=1, pretrained=False, block_type='upsampling', concat_input=True, trainable=False
                          , framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True
                          , dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B2 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B2 model
    """
    # encoder, block_lookup = get_efficientnet_b2_encoder(input_shape, pretrained=pretrained)
    # model = _get_efficient_unet(encoder, n_contour_classes, block_type=block_type, concat_input=concat_input, block_lookup=block_lookup)
    # return model

    encoder, block_lookup = get_efficientnet_b2_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, n_celltype_classes,block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model


def get_efficient_unet_b3(input_shape, n_contour_classes=3, n_celltype_classes=1, pretrained=False, block_type='upsampling', concat_input=True, trainable=False
                          , framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True
                          , dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B3 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B3 model
    """
    # encoder, block_lookup = get_efficientnet_b3_encoder(input_shape, pretrained=pretrained)
    # model = _get_efficient_unet(encoder, n_contour_classes, block_type=block_type, concat_input=concat_input, block_lookup=block_lookup)
    # return model

    encoder, block_lookup = get_efficientnet_b3_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, n_celltype_classes,block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model

def get_efficient_unet_b4(input_shape, n_contour_classes=3, n_celltype_classes=1, pretrained=False, block_type='upsampling', concat_input=True, trainable=False
                          , framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True
                          , dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B4 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B4 model
    """
    # encoder, block_lookup = get_efficientnet_b4_encoder(input_shape, pretrained=pretrained)
    # model = _get_efficient_unet(encoder, n_contour_classes, block_type=block_type, concat_input=concat_input, block_lookup=block_lookup)
    # return model

    encoder, block_lookup = get_efficientnet_b4_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, n_celltype_classes,block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model

def get_efficient_unet_b5(input_shape, n_contour_classes=3, pretrained=False, block_type='upsampling', concat_input=True, trainable=False, framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True, dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B5 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :param trainable: True for encoder with trainable weights
    :param attention: True for attention gates between decoder paths
    :param dist_regression: False for removing decoder regressing gradient maps (all)
    :param diag_dist: True for having only ho-ver gradient maps with no attention (similarly to original hover-net model) 
    :return: an EfficientUnet_B5 model

    attention true: 
        dist_regression true
        diag_dist true
        --> 1.full model with attention
        dist_regression true
        diag_dist false
        --> 1.hover model with attention
    attention false
            dist_regression true
            diag_dist true
            --> 2.full model without attention
            dist_regression true
            diag_dist false
            --> 3.only hover without attention (ignored for ablation)
            dist_regression false
            --> 4.only pixel classification (without attention)
    """

    encoder, block_lookup = get_efficientnet_b5_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model

def get_efficient_unet_b6(input_shape, n_contour_classes=3, pretrained=False, block_type='upsampling', concat_input=True, trainable=False, framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True, dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B6 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B6 model
    """
    # encoder, block_lookup = get_efficientnet_b6_encoder(input_shape, pretrained=pretrained)
    # model = _get_efficient_unet(encoder, n_contour_classes, block_type=block_type, concat_input=concat_input, block_lookup=block_lookup)
    # return model

    encoder, block_lookup = get_efficientnet_b6_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model


def get_efficient_unet_b7(input_shape, n_contour_classes=3, pretrained=False, block_type='upsampling', concat_input=True, trainable=False, framework = False, encdec_skipconn_attention = True, decdec_skipconn_attention = True, dist_regression = True, diag_dist = True, skipp_conn_decod = True):
    """Get a Unet model with Efficient-B7 encoder
    :param input_shape: shape of input (cannot have None element)
    :param n_contour_classes: the number of output channels
    :param pretrained: True for ImageNet pretrained weights
    :param block_type: "upsampling" to use UpSampling layer, otherwise use Conv2DTranspose layer
    :param concat_input: if True, input image will be concatenated with the last conv layer
    :return: an EfficientUnet_B7 model
    """
    # encoder, block_lookup = get_efficientnet_b7_encoder(input_shape, pretrained=pretrained)
    # model = _get_efficient_unet(encoder, n_contour_classes, block_type=block_type, concat_input=concat_input, block_lookup=block_lookup)
    # return model

    encoder, block_lookup = get_efficientnet_b7_encoder(input_shape, pretrained=pretrained)
    model = _get_efficient_unet(encoder, block_lookup, n_contour_classes, block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skipp_conn_decod)
    return model
    
def _get_efficient_unet(encoder, block_lookup, n_contour_classes, n_celltype_classes, block_type, concat_input, trainable, framework, encdec_skipconn_attention, decdec_skipconn_attention, dist_regression, diag_dist, skip_conn_dec):  
    if not trainable:
        for layer in encoder.layers:
            layer.trainable=False
    if framework:
        print("building model FRAMEWORK")
        model = _get_efficient_unet_framework(encoder, block_lookup, n_contour_classes=3, block_type=block_type, concat_input=True)
    if n_contour_classes >= 3:
        if encdec_skipconn_attention and decdec_skipconn_attention:
            if dist_regression and diag_dist and skip_conn_dec: # default full model with attention
                print("building model DEFAULT")
                #_get_CISCA_unet3_reduced_squeezeexcite _get_CISCA_unet3_reduced
                model = _get_CISCA_unet3_reduced_squeezeexcite(encoder, block_lookup, n_contour_classes, n_celltype_classes, block_type=block_type, concat_input=concat_input)
            elif dist_regression and not diag_dist and skip_conn_dec: # similar to hover model with attention (nodiag)
                print("building model NODIAG")
                model = _get_CISCA_unet3_reduced_squeezeexcite(encoder, block_lookup, n_contour_classes, n_celltype_classes, block_type=block_type, concat_input=True, grad_channels=2)     
            else:
                print("error in the flags :))")
        elif not encdec_skipconn_attention and not decdec_skipconn_attention:
            if dist_regression and diag_dist and skip_conn_dec: # full model without attention, skip connections included (noatt)
                print("building model NOATT")
                model = _get_efficient_unet_noatt(encoder, block_lookup, n_contour_classes, block_type=block_type, concat_input=True)
            elif dist_regression and not diag_dist and skip_conn_dec: # no diag, no att in enc, no att in dec (nodiagnoatt)
                print("building model NODIAGNOATT")
                model = _get_efficient_unet_noatt(encoder, block_lookup, n_contour_classes, grad_channels=2, block_type=block_type, concat_input=True)
            elif dist_regression and diag_dist and not skip_conn_dec: # full model with no skip connections between decoders, no att (noattnoskip)  
                print("building model NOATTNOSKIP")
                model = _get_efficient_unet_noskipdecoder(encoder, block_lookup, n_contour_classes, block_type=block_type, concat_input=concat_input, attention=None)  
            elif dist_regression and not diag_dist and not skip_conn_dec: # no diag with no skip connections between decoders, no att (nodiagnoattnoskip)   
                print("building model NODIAGNOATTNOSKIP")
                model = _get_efficient_unet_noskipdecoder(encoder, block_lookup, n_contour_classes, grad_channels=2, block_type=block_type, concat_input=concat_input, attention=None)  
            elif not dist_regression: # only pixel classification (without attention) (nograd)
                print("building model NOGRAD")
                model = _get_efficient_unet_nograd(encoder, block_lookup, n_contour_classes, block_type=block_type, concat_input=True, attention = None)
            else:
                print("error in the flags :))")
        elif encdec_skipconn_attention and not decdec_skipconn_attention:
            if dist_regression and not diag_dist and not skip_conn_dec: # no diag, att in encod, no skip in dec (nodiagattnoskip)
                print("building model NODIAGATTNOSKIP")
                model = _get_efficient_unet_noskipdecoder(encoder, block_lookup, n_contour_classes, grad_channels=2, block_type=block_type, concat_input=concat_input)  
            elif dist_regression and not diag_dist and skip_conn_dec: # no diag, att in encod, no att in dec (nodiagattnoatt)
                print("building model NODIAGATTNOATT")
                model = _get_efficient_unet_noatt(encoder, block_lookup, n_contour_classes, grad_channels=2, block_type=block_type, concat_input=True, attention="add")
            elif dist_regression and diag_dist and skip_conn_dec: # full model, att in encod, no att in dec (attnoatt)
                print("building model ATTNOATT")
                model = _get_efficient_unet_noatt(encoder, block_lookup, n_contour_classes, block_type=block_type, concat_input=True, attention="add")
            elif dist_regression and diag_dist and not skip_conn_dec: # full model with no skip connections between decoders (attnoskip)
                print("building model ATTNOSKIP")
                model = _get_efficient_unet_noskipdecoder(encoder, block_lookup, n_contour_classes, block_type=block_type, concat_input=concat_input)
            elif not dist_regression: # only pixel classification (with attention) (attnograd)
                print("building model ATTNOGRAD")
                model = _get_efficient_unet_nograd(encoder, block_lookup, n_contour_classes, block_type=block_type, concat_input=True)
            else:
                print("error in the flags :))")
        elif not encdec_skipconn_attention and decdec_skipconn_attention:
            if dist_regression and diag_dist and skip_conn_dec: # full model, no att in enc, att in dec (noattatt)
                print("building model NOATTATT")
                model = _get_CISCA_unet3(encoder, block_lookup, n_contour_classes, block_type=block_type, concat_input=True, attention = None)
            elif dist_regression and not diag_dist and skip_conn_dec: # no diag, no att in enc, att in dec (nodiagnoattatt)
                print("building model NODIAGNOATTATT")
                model = _get_CISCA_unet3(encoder, block_lookup, n_contour_classes, grad_channels=2, block_type=block_type, concat_input=True, attention = None)
            else:
                print("error in the flags :))")
    else:
        if encdec_skipconn_attention and decdec_skipconn_attention:
            if dist_regression and diag_dist and skip_conn_dec: # default full model with attention
                print("building model DEFAULTBW")
                model = _get_CISCA_unet3(encoder, block_lookup, n_contour_classes=1, block_type=block_type, concat_input=concat_input, activationlast='sigmoid')
            elif dist_regression and not diag_dist and skip_conn_dec: # similar to hover model with attention (nodiag)
                print("building model NODIAGBW")
                model = _get_CISCA_unet3(encoder, block_lookup, n_contour_classes=1, block_type=block_type, concat_input=True, grad_channels=2, activationlast='sigmoid')     
            else:
                print("error in the flags :))")
    return model

def attention_gate(X, g, channel,  
                   activation='relu', 
                   attention='add', name='att'):
    '''
    Self-attention gate modified from Oktay et al. 2018.
    
    attention_gate(X, g, channel,  activation='ReLU', attention='add', name='att')
    
    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X_att: output tensor.
    
    '''
    #activation_func = eval(activation)
    attention_func = eval(attention)
    
    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)
    
    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)
    
    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))
    
    # nonlinear activation
    f = Activation(activation,name='{}_activation'.format(name))(query)
    #f = activation_func(name='{}_activation'.format(name))(query)
    
    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #
    
    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)
    
    # multiplicative attention masking
    X_att = multiply([X, coef_att], name='{}_masking'.format(name))
    
    return X_att

def CONV_stack(X, channel, kernel_size=3, stack_num=2, 
            dilation_rate=1, activation='ReLU', 
            batch_norm=False, name='conv_stack'):
    '''
    Stacked convolutional layers:
    (Convolutional layer --> batch normalization --> Activation)*stack_num

    CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU', 
                batch_norm=False, name='conv_stack')


    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked Conv2D-BN-Activation layers.
        dilation_rate: optional dilated convolution kernel.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''

    bias_flag = not batch_norm

    # stacking Convolutional layers
    for i in range(stack_num):
        
        activation_func = eval(activation)
        
        # linear convolution
        X = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag, 
                    dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)
        
        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)
    
    return X


