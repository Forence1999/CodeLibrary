import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Reshape, Conv2D, \
    BatchNormalization
from tensorflow.keras.models import Sequential


class ResCNN_4_STFT_DOA(tf.keras.Model):  # channel-last
    def __init__(self, num_classes, num_time_clips, num_res_block=2, num_filter=32, name=None, **kwargs):
        name = name if (name is not None) else self.__class__.__name__
        super(ResCNN_4_STFT_DOA, self).__init__(name=name, **kwargs)
        
        self.conv_1 = Sequential([
            Conv2D(filters=num_filter * 2, kernel_size=(1, 7), strides=(1, 3), padding='valid', ),  # (32, 7, 110)
            BatchNormalization(axis=-1), Activation('relu'),
            Conv2D(filters=num_filter, kernel_size=(1, 5), strides=(1, 2), padding='valid', ),  # (32, 7, 52)
            BatchNormalization(axis=-1), Activation('relu')]
        )
        self.res_block_ls = []
        for _ in range(num_res_block):
            res_conv = Sequential([
                Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
                BatchNormalization(axis=-1), Activation('relu'),
                Conv2D(num_filter, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(axis=-1), Activation('relu'),
                Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
                BatchNormalization(axis=-1)])
            self.res_block_ls.append(res_conv)
        if num_res_block > 0:
            self.relu = Activation('relu')
        
        self.conv_2 = Sequential([
            Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=-1), Activation('relu')], name='feature')
        
        self.conv_3 = Sequential([
            Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=-1), Activation('relu')])
        
        self.conv_4 = Sequential([
            Conv2D(1, kernel_size=(num_time_clips, 1), strides=(1, 1), padding='valid', use_bias=True),
            BatchNormalization(axis=-1), Reshape((-1,)), Activation('softmax')])
        # self.flatten = Flatten(name='flatten')
    
    def call(self, x, training=None, mask=None):
        # Input.shape: (None, 30, 508, 8)
        x = self.conv_1(x)  # (None, 30, 82, 64)
        for res_block in self.res_block_ls:
            x = self.relu(x + res_block(x))  # (None, 30, 82, 64)
        x0 = self.conv_2(x)  # (None, 30, 82, 8)
        x = K.permute_dimensions(x0, (0, 1, 3, 2))  # (None, 30, 8, 82)
        x = self.conv_3(x)  # (None, 30, 12, 32)
        x = self.conv_4(x)  # (None, 8)
        # x = self.flatten(x)
        return x


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    num_clips = 30
    stft_len = 508
    model = ResCNN_4_STFT_DOA(num_classes=8, num_time_clips=num_clips, num_res_block=2,
                              num_filter=64)
    model.build(input_shape=(None, num_clips, stft_len, 8,))
    model.summary()
    # rand_input = np.random.random((3, 8, 7, 508))
    # y = model(rand_input)
    # print('y:', y.numpy())
    print('Hello World!')
