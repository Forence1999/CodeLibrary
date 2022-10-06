import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Reshape, Conv2D, BatchNormalization
from tensorflow.keras.models import Sequential


class ResCNN_4_STFT_DOA(tf.keras.Model):  # channel-first
    def __init__(self, num_classes, H, num_res_block=2, num_filter=32, name=None, **kwargs):
        name = name if (name is not None) else self.__class__.__name__
        super(ResCNN_4_STFT_DOA, self).__init__(name=name, **kwargs)
        print('-' * 20, 'This is a channel-first model.', '-' * 20, )
        
        # self.model_input = Input(shape=(8, 7, 337))
        self.num_res_block = num_res_block
        self.conv_1 = Sequential([
            Conv2D(filters=num_filter * 2, kernel_size=(1, 7), strides=(1, 3), padding='valid', ),  # (32, 7, 110)
            BatchNormalization(axis=1), Activation('relu'),
            Conv2D(filters=num_filter, kernel_size=(1, 5), strides=(1, 2), padding='valid', ),  # (32, 7, 52)
            BatchNormalization(axis=1), Activation('relu')]
        )
        self.res_block_ls = []
        for _ in range(self.num_res_block):
            res_conv = Sequential([
                Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
                BatchNormalization(axis=1), Activation('relu'),
                Conv2D(num_filter, kernel_size=(3, 3), strides=(1, 1), padding='same'),
                BatchNormalization(axis=1), Activation('relu'),
                Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='same'),
                BatchNormalization(axis=1)])
            self.res_block_ls.append(res_conv)
        if self.num_res_block > 0:
            self.relu = Activation('relu')
        
        self.conv_2 = Sequential([
            Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=1), Activation('relu')])
        
        self.conv_3 = Sequential([
            Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=1), Activation('relu')])
        
        self.conv_4 = Sequential([
            Conv2D(1, kernel_size=(H, 1), strides=(1, 1), padding='valid'),
            BatchNormalization(axis=1),
            Reshape((-1,)), Activation('softmax')])
        # self.flatten = Flatten(name='flatten')
    
    def call(self, x, training=None, mask=None):
        # a.shape: [B, 8, 7, 337]
        # inputs = self.model_input(inputs)
        x = self.conv_1(x)  # [B, 128, 7, 54]
        for res_block in self.res_block_ls:
            x = self.relu(x + res_block(x))  # [B, 128, 7, 54]
        x0 = self.conv_2(x)  # [B, 360, 7, 54]
        x = K.permute_dimensions(x0, (0, 3, 2, 1))  # [B, 54, 7, 360]
        x = self.conv_3(x)  # [B, 500, 7, 360]
        x = self.conv_4(x)  # [B, 1, 1, 360]
        # x = self.flatten(x)
        return x


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    K.set_image_data_format('channels_first')
    # for i in range(6):
    #     print('\n', 'Number of ResConv Blocks:', i, '\n', )
    model = ResCNN_4_STFT_DOA(num_classes=8, H=30, num_res_block=2, num_filter=64)  # 4 * 2 ** i)
    model.build(input_shape=(None, 8, 30, 508))
    model.summary()
    # rand_input = np.random.random((3, 8, 7, 508))
    # y = model(rand_input)
    # print('y:', y.numpy())
    print('Hello World!')
