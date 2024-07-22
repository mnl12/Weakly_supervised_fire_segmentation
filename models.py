import tensorflow as tf
import numpy as np

class feat_thre(tf.keras.layers.Layer):
    def __init__(self, thre):
        super(feat_thre, self).__init__(name='segment_out')
        self.thre = thre

    def call(self, inputs):
        max_v=tf.reduce_max(inputs, axis=[1,2], keepdims=True)
        return tf.where(tf.less(inputs, tf.multiply(self.thre, max_v)), tf.zeros_like(inputs), tf.ones_like(inputs))


class sum_feat(tf.keras.layers.Layer):
    def __init__(self):
        super(sum_feat, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=-1, keepdims=True)

class weight_avg(tf.keras.layers.Layer):
    def __init__(self):
        super(weight_avg, self).__init__()
        w_init = tf.zeros_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(1,), dtype="float32"), trainable=True)

    def call(self, inputs):
        return tf.multiply(self.w, inputs)

class normalize_feat(tf.keras.layers.Layer):
    def __init__(self):
        super(normalize_feat, self).__init__()

    def call(self, inputs):
        [normout,normv]=tf.linalg.normalize(inputs, axis=[1,2])
        return normout


class const_weight_avg(tf.keras.layers.Layer):
    def __init__(self, dim_w):
        super(const_weight_avg, self).__init__()
        self.initw= tf.constant(1, shape=(dim_w,), dtype="float32")
        self.w = tf.tensor_scatter_nd_add(self.initw, [[0]], [30])

    def call(self, inputs):
        return tf.multiply(self.w, inputs)

class scale_inv_avg_pool(tf.keras.layers.Layer):
    def __init__(self,thre):
        super(scale_inv_avg_pool, self).__init__()
        self.thre = thre

    def call(self, inputs):
        max_input = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        input_thre = tf.where(tf.less(inputs, tf.multiply(tf.multiply(self.thre, max_input), tf.ones_like(inputs))),
                              tf.zeros_like(inputs), inputs)
        num_nonzeros = tf.math.count_nonzero(input_thre, axis=[1, 2], dtype=tf.float32)
        num_nonzeros = tf.math.add(num_nonzeros, tf.ones_like(num_nonzeros))
        out = tf.math.divide(tf.reduce_sum(input_thre, axis=[1, 2]), num_nonzeros)
        return out




def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = tf.keras.layers.Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y


def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = tf.keras.backend.int_shape(tensor)

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = tf.keras.layers.Activation('relu', name=f'relu_1')(y_pool)

   # y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])
    y_pool = tf.keras.layers.UpSampling2D((dims[1], dims[2]), interpolation='bilinear')(y_pool)

    y_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = tf.keras.layers.BatchNormalization(name=f'bn_2')(y_1)
    y_1 = tf.keras.layers.Activation('relu', name=f'relu_2')(y_1)

    y_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = tf.keras.layers.BatchNormalization(name=f'bn_3')(y_6)
    y_6 = tf.keras.layers.Activation('relu', name=f'relu_3')(y_6)

    y_12 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = tf.keras.layers.BatchNormalization(name=f'bn_4')(y_12)
    y_12 = tf.keras.layers.Activation('relu', name=f'relu_4')(y_12)

    y_18 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = tf.keras.layers.BatchNormalization(name=f'bn_5')(y_18)
    y_18 = tf.keras.layers.Activation('relu', name=f'relu_5')(y_18)

    y = tf.keras.layers.concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(name=f'bn_final')(y)
    y = tf.keras.layers.Activation('relu', name=f'relu_final')(y)
    return y

def label_supervised(tensor, OUTPUT_CHANNELS):
    g1=tf.keras.layers.GlobalAveragePooling2D()(tensor)
    g1 = tf.keras.layers.Dense(256, activation='relu')(g1)
    g1=tf.keras.layers.Dense(OUTPUT_CHANNELS, activation=None)(g1)
    return g1



def segmentation_network (base_model_name,decoder_name, n_classes, IMAGE_SIZE, add_feat):
    OUTPUT_CHANNELS = n_classes
    if base_model_name=='mobilenet':
        base_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
    elif base_model_name=='vgg':
        base_model = tf.keras.applications.VGG16(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        layer_names = [
            'block1_pool',  # 64
            'block2_pool',
            'block3_pool',
            'block4_conv3',
            'block5_conv3'
        ]
    elif base_model_name == 'resnet':
        base_model = tf.keras.applications.ResNet50(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        layer_names = ['conv2_block3_out', 'conv4_block6_out']



    #layers = base_model.get_layer(layer_names[-1]).output
    layers = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = True

    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3])
    enc_outs = down_stack(inputs)



    if decoder_name =='fcn-32':
        y1=enc_outs[-1]
        y1=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1,1), use_bias=False, kernel_initializer='zeros')(y1)
        pool_size = tuple(imsize // 32 for imsize in IMAGE_SIZE)
        g = tf.keras.layers.GlobalAvgPool2D()(y1)
        g = tf.keras.layers.Flatten()(g)
        g = tf.keras.layers.Activation('sigmoid', name='class_out')(g)
        y=tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(y1)
        y_sig=tf.keras.layers.Activation('sigmoid')(y)


    elif decoder_name =='fcn-16':
        y1=enc_outs[-1]
        y2=enc_outs[-2]
        y1=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1,1), use_bias=False, kernel_initializer='zeros')(y1)
        g = tf.keras.layers.GlobalAvgPool2D()(y1)
        g = tf.keras.layers.Flatten()(g)
        g = tf.keras.layers.Activation('sigmoid', name='class_out')(g)
        y=tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(y1)
        #y_sig=tf.keras.layers.Activation('sigmoid')(y)
        y_sig=sum_feat()(y2)
        y_sig = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(y_sig)

    elif decoder_name=='fcn-consist':
        y1 = enc_outs[-1]
        y2 = enc_outs[-2]
        y1 = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1, 1), use_bias=False, kernel_initializer='zeros')(y1)
        g = tf.keras.layers.GlobalAvgPool2D()(y1)
        g = tf.keras.layers.Flatten()(g)
        g = tf.keras.layers.Activation('sigmoid', name='class_out')(g)
        y = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(y1)
        #y_sig = tf.keras.layers.Activation('sigmoid')(y)
        y2 = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1, 1), use_bias=False, kernel_initializer='zeros')(y2)
        y_sig = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(y2)

    elif decoder_name=='deeplab':
        y=ASPP(enc_outs[-1])
        y=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 1, padding='same', activation=None)(y)
        y=tf.keras.layers.UpSampling2D((16, 16), interpolation='bilinear', name='upsampled')(y)
        y_sig=tf.keras.layers.Activation('sigmoid')(y)
        g=tf.keras.layers.GlobalAvgPool2D()(y)


    return tf.keras.Model(inputs=inputs, outputs=[g, y, y_sig])



