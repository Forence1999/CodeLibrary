from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model


def FCN(num_classes, Chans=64, SamplePoints=128, dropoutRate=None, norm_rate=0.25, ):
    input1 = Input(shape=(1, Chans, SamplePoints))
    block1 = Flatten()(input1)
    
    block1 = Dense(128, activation='relu', kernel_constraint=max_norm(norm_rate))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = AveragePooling2D((1, 16))(block1)
    if dropoutRate is not None:
        block1 = Dropout(dropoutRate)(block1)
    
    block2 = Dense(128, activation='relu', kernel_constraint=max_norm(norm_rate))(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    if dropoutRate is not None:
        block2 = Dropout(dropoutRate)(block2)
    
    block3 = Dense(128, activation='relu', kernel_constraint=max_norm(norm_rate))(block2)
    block3 = BatchNormalization(axis=1)(block3)
    block3 = AveragePooling2D((1, 8))(block3)
    if dropoutRate is not None:
        block3 = Dropout(dropoutRate)(block3)
    
    block4 = Dense(num_classes, activation='relu', kernel_constraint=max_norm(norm_rate))(block3)
    softmax = Activation('softmax', name='softmax')(block4)
    
    return Model(inputs=input1, outputs=softmax)


# class FCN(tf.keras.Model):
#
#     def __init__(self, num_classes, name=None, **kwargs):
#         if name is None:
#             name = self.__class__.__name__
#
#         super(FCN, self).__init__(name=name, **kwargs)
#
#         self.flatten = Flatten()
#         self.dense1 = Dense(256, activation='relu')
#         self.dense2 = Dense(512, activation='relu')
#         self.out = Dense(num_classes, activation='linear')
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.flatten(inputs)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         out = self.out(x)
#
#         if training is False:
#             out = tf.nn.softmax(out, axis=-1)
#
#         return out
#


if __name__ == '__main__':
    
    print('Hello World!')
