import os
import sys

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Activation, Dropout, Reshape, Conv2D, BatchNormalization, PReLU
from tensorflow.keras.models import Sequential


def ResCNN_4_STFT_DOA(num_classes, num_time_clips, num_res_block=2, stft_len=508, num_filter=32, dropoutRate=0.,
                      name=None):
    name = sys._getframe().f_code.co_name if (name is None) else name

    input = Input(shape=(num_time_clips, stft_len, 8), name='input')
    conv_1 = Sequential([
        Conv2D(filters=num_filter * 2, kernel_size=(1, 7), strides=(1, 3), padding='valid', ),
        BatchNormalization(axis=-1), Activation('relu'),
        Conv2D(filters=num_filter, kernel_size=(1, 5), strides=(1, 2), padding='valid', ),
        BatchNormalization(axis=-1), Activation('relu')],
        name='input_conv')
    res_block_ls = []
    relu_ls = []
    for i in range(num_res_block):
        res_conv = Sequential([
            Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1), Activation('relu'),
            Conv2D(num_filter, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1), Activation('relu'),
            Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1)],
            name='res_block_%d' % i)
        res_block_ls.append(res_conv)
        relu_ls.append(Activation('relu', name='relu_%d' % i))

    conv_2 = Sequential([
        Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
        BatchNormalization(axis=-1), Activation('relu')], name='feature')

    conv_3 = Sequential([
        Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
        BatchNormalization(axis=-1), Activation('relu')],
        name='feature_conv')

    conv_4 = Sequential([
        Conv2D(1, kernel_size=(num_time_clips, 1), strides=(1, 1), padding='valid', use_bias=True),
        Reshape((-1,)), ],
        name='output_conv')  # BatchNormalization(axis=-1),
    softamx = Activation('softmax', name='softmax')
    # flatten = Flatten(name='flatten')

    # call
    # Input.shape: (None, 30, 508, 8)
    x = conv_1(input)  # (None, 30, 82, 64)
    for relu_i, res_block_i in zip(relu_ls, res_block_ls):
        x = relu_i(x + res_block_i(x))  # (None, 30, 82, 64)
    x = conv_2(x)  # (None, 30, 82, 8)
    x = K.permute_dimensions(x, (0, 1, 3, 2))  # (None, 30, 8, 82)
    x = Activation('linear', name='feature_linear')(x)
    # dropout
    x = Dropout(rate=dropoutRate, noise_shape=(None, 30, 1, 82,), )(x)
    # x = conv_3(x)  # (None, 30, 12, 32)
    x = Activation('linear', name='feature_conv_linear')(x)
    x = conv_4(x)  # (None, 8)
    x = Activation('linear', name='output_linear')(x)
    output = softamx(x)  # (None, 8)
    # x = flatten(x)

    return tf.keras.Model(inputs=input, outputs=output, name=name)


def ResCNN_4_STFT_DOA_PReLU(num_classes, num_time_clips, num_res_block=2, stft_len=508, num_filter=32, dropoutRate=0.25,
                            name=None):
    name = sys._getframe().f_code.co_name if (name is None) else name

    input = Input(shape=(num_time_clips, stft_len, 8), name='input')
    conv_1 = Sequential([
        Conv2D(filters=num_filter * 2, kernel_size=(1, 7), strides=(1, 3), padding='valid', ),
        BatchNormalization(axis=-1), PReLU(shared_axes=(1, 2, 3), ),
        Conv2D(filters=num_filter, kernel_size=(1, 5), strides=(1, 2), padding='valid', ),
        BatchNormalization(axis=-1), PReLU(shared_axes=(1, 2, 3), )],
        name='input_conv')
    res_block_ls = []
    relu_ls = []
    for i in range(num_res_block):
        res_conv = Sequential([
            Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1), PReLU(shared_axes=(1, 2, 3), ),
            Conv2D(num_filter, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1), PReLU(shared_axes=(1, 2, 3), ),
            Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1)],
            name='res_block_%d' % i)
        res_block_ls.append(res_conv)
        relu_ls.append(PReLU(shared_axes=(1, 2, 3), name='prelu_%d' % i))

    conv_2 = Sequential([
        Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
        BatchNormalization(axis=-1), PReLU(shared_axes=(1, 2, 3), )], name='feature')

    conv_3 = Sequential([
        Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
        BatchNormalization(axis=-1), PReLU(shared_axes=(1, 2, 3), )],
        name='feature_conv')

    conv_4 = Sequential([
        Conv2D(1, kernel_size=(num_time_clips, 1), strides=(1, 1), padding='valid', use_bias=True),
        BatchNormalization(axis=-1), Reshape((-1,)), ],
        name='output_conv')
    softamx = Activation('softmax', name='softmax')
    # flatten = Flatten(name='flatten')

    # call
    # Input.shape: (None, 30, 508, 8)
    x = conv_1(input)  # (None, 30, 82, 64)
    for relu_i, res_block_i in zip(relu_ls, res_block_ls):
        x = relu_i(x + res_block_i(x))  # (None, 30, 82, 64)
    x = conv_2(x)  # (None, 30, 82, 8)
    x = K.permute_dimensions(x, (0, 1, 3, 2))  # (None, 30, 8, 82)
    x = Activation('linear', name='feature_linear')(x)
    # dropout
    x = Dropout(rate=dropoutRate, noise_shape=(None, 30, 1, 82,), )(x)
    # x = conv_3(x)  # (None, 30, 12, 32)
    # x = Activation('linear', name='feature_conv_linear')(x)
    x = conv_4(x)  # (None, 8)
    x = Activation('linear', name='output_linear')(x)
    output = softamx(x)  # (None, 8)
    # x = flatten(x)

    return tf.keras.Model(inputs=input, outputs=output, name=name)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_clips = 30
    stft_len = 508
    model = ResCNN_4_STFT_DOA(num_classes=8, num_time_clips=num_clips, stft_len=stft_len, num_res_block=2,
                              num_filter=64, )
    model.build(input_shape=(None, num_clips, stft_len, 8,))
    model.summary()
    # rand_input = np.random.random((3, 8, 7, 508))
    # y = model(rand_input)
    # print('y:', y.numpy())
    print('Hello World!')
