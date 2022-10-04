# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022

from pathlib import Path

import numpy as np
import tensorflow as tf


def CYC_PathDatasetGenerator(src_paths, ):
    x_paths = []
    for src_path in src_paths:
        x_paths.extend(Path(src_path).rglob(pattern='*.npz'))
    y_s = [int(i.parts[-3]) // 45 for i in x_paths]
    x_paths = [str(i) for i in x_paths]
    
    return list(zip(x_paths, y_s))


def CYC_map_func(x_path, y):
    x = np.load(x_path.numpy())['data'][np.newaxis, :]
    
    return (x, y)


def general_function(func, inp, Tout):
    x, y = tf.py_function(func=func, inp=inp, Tout=Tout)
    x.set_shape(tf.TensorShape([1, 6, 128]))
    y.set_shape(tf.TensorShape(()))
    return x, y


def CYC_DataGenerator(path_ds, batch_size=None, batch_drop_remainder=False, map_parallel_calls=None,
                      map_deterministic=None, shuffle_buffer_size=1024, shuffle_seed=None,
                      reshuffle_each_iteration=True, num_prefetch=1, ):
    '''
    return a data generator for CYC dataset based on a list of (x_path, y) pair
    Args:
        path_ds:
    Returns: a data generator for CYC dataset
    '''
    xs, ys = list(zip(*path_ds))
    xs, ys = tf.constant(xs), tf.constant(ys)
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed,
                              reshuffle_each_iteration=reshuffle_each_iteration)
    dataset = dataset.map(
        map_func=lambda x_path, y: general_function(func=CYC_map_func, inp=(x_path, y), Tout=(tf.float64, tf.int32)),
        # map_func=lambda x_path, y: tf.py_function(func=CYC_map_func, inp=[x_path, y], Tout=(tf.float64, tf.int32)),
        num_parallel_calls=map_parallel_calls, deterministic=map_deterministic)
    
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=batch_drop_remainder)
    dataset = dataset.prefetch(num_prefetch)
    return dataset


if __name__ == '__main__':
    x = list(range(10))
    y = list(range(10))
    path_ds = list(zip(x, y))
    
    CYC_DataGenerator(path_ds)
    
    print('Hello World!')
