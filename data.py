# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022

import os
import pickle
import random
from collections import Counter

import numpy as np


EPS = np.finfo(np.float32).eps


# -------------------------- normalization -------------------------- #

def standard_normalizaion(x):
	return (x - np.mean(x)) / (np.std(x) + EPS)


def standard_normalization_wise(data, normalization=None):  # TODO: has not been tested
	''' channel-first '''
	data = np.array(data)
	if normalization is None:
		return data
	assert normalization in ['whole', 'sample-wise', 'channel-wise', 'samplepoint-wise']
	for i in range(len(data)):
		if normalization == 'whole':
			data = standard_normalizaion(data)
		elif normalization == 'sample-wise':
			data[i, :, :] = standard_normalizaion(data[i, :, :])
		elif normalization == 'channel-wise':
			data[i, :, :] = [standard_normalizaion(data[i, j, :]) for j in range(data.shape[-2])]
		elif normalization == 'samplepoint-wise':
			data[i, :, :] = np.array([standard_normalizaion(data[i, :, j]) for j in range(data.shape[-1])]).T
		else:
			print('-' * 20, 'normalization is incorrectly assigned', '-' * 20)

	return np.array(data)


# -------------------------- split -------------------------- #

def split_data(data, split=0.8, shuffle=True):  # TODO: has not been tested
	x = data[0]
	y = data[1]
	data_size = len(x)
	split_index = int(data_size * split)
	indices = np.arange(data_size)
	if shuffle:
		indices = np.random.permutation(indices)
	x_train = x[indices[:split_index]]
	y_train = y[indices[:split_index]]
	x_test = x[indices[split_index:]]
	y_test = y[indices[split_index:]]
	return x_train, y_train, x_test, y_test


def split_data_wid(data, split=0.8, shuffle=True):  # TODO: has not been tested
	x = data[0]
	y = data[1]
	s = data[2]
	data_size = len(x)
	split_index = int(data_size * split)
	indices = np.arange(data_size)
	if shuffle:
		indices = np.random.permutation(indices)
	x_train = x[indices[:split_index]]
	y_train = y[indices[:split_index]]
	s_train = s[indices[:split_index]]
	x_test = x[indices[split_index:]]
	y_test = y[indices[split_index:]]
	return x_train, y_train, s_train, x_test, y_test


def split_data_both(data, split=0.8, shuffle=True):  # TODO: has not been tested
	x = data[0]
	x_poison = data[1]
	y = data[2]
	s = data[3]
	data_size = len(x)
	split_index = int(data_size * split)
	indices = np.arange(data_size)
	if shuffle:
		indices = np.random.permutation(indices)
	x_train = x[indices[:split_index]]
	x_train_poison = x_poison[indices[:split_index]]
	y_train = y[indices[:split_index]]
	s_train = s[indices[:split_index]]
	x_test = x[indices[split_index:]]
	y_test = y[indices[split_index:]]
	return x_train, x_train_poison, y_train, s_train, x_test, y_test


def get_split_indices(data_size, split=[9, 1], shuffle=True):  # TODO: has not been tested
	if len(split) < 2:
		raise TypeError(
			'The length of split should be larger than 2 while the length of your split is {}!'.format(len(split)))
	split = np.array(split)
	split = split / np.sum(split)
	if shuffle:
		indices = get_shuffle_index(data_size)
	else:
		indices = np.arange(data_size)
	split_indices_list = []
	start = 0
	for i in range(len(split) - 1):
		end = start + int(np.floor(split[i] * data_size))
		split_indices_list.append(indices[start:end])
		start = end
	split_indices_list.append(indices[start:])
	return split_indices_list


def gen_cross_val_idx(num_sbj, num_fold, num_subfold, random_seed=None, ):  # TODO: has not been tested
	if random_seed is not None:
		os.environ["PYTHONHASHSEED"] = str(random_seed)
		random.seed(random_seed)
		np.random.seed(random_seed)

	sbj_rand_idx = get_shuffle_index(num_sbj)
	split_ds_idx = [sbj_rand_idx[i::num_fold] for i in range(num_fold)] * num_fold

	all_split_fold = []
	for i in range(num_fold):
		train_idx = split_ds_idx[i:i + num_subfold[0]]
		val_idx = split_ds_idx[i + num_subfold[0]:i + num_subfold[0] + num_subfold[1]]
		test_idx = split_ds_idx[
				   i + num_subfold[0] + num_subfold[1]:i + num_subfold[0] + num_subfold[1] + num_subfold[2]]
		train_idx = sorted(np.concatenate(train_idx) if num_subfold[0] > 1 else train_idx[0])  #
		val_idx = sorted(np.concatenate(val_idx) if num_subfold[1] > 1 else val_idx[0])  #
		test_idx = sorted(np.concatenate(test_idx) if num_subfold[2] > 1 else test_idx[0])  #
		all_split_fold.append([train_idx, val_idx, test_idx])
	return all_split_fold


# -------------------------- shuffle -------------------------- #

def shuffle_data(data, random_seed=None):  # TODO: has not been tested
	'''
	data: [x, y]   type: numpy
	'''
	x, y = data
	data_size = x.shape[0]
	shuffle_index = get_shuffle_index(data_size, random_seed=random_seed)

	return x[shuffle_index], y[shuffle_index]


def get_shuffle_index(data_size, random_seed=None):  # TODO: has not been tested
	if random_seed is not None:
		np.random.seed(random_seed)
	return np.random.permutation(np.arange(data_size))


# -------------------------- iterate -------------------------- #

def batch_iter(data, batchsize, shuffle=True, random_seed=None):  # TODO: has not been tested
	# Example: batches = list(utils.batch_iter([x_train, y_train], batchsize=batchsize, shuffle=True, random_seed=None))

	'''split dataset into batches'''
	if shuffle:
		x, y = shuffle_data(data, random_seed=random_seed)
	else:
		x, y = data
	data_size = x.shape[0]
	nb_batches = np.ceil(data_size / batchsize).astype(np.int)

	for batch_id in range(nb_batches):
		start_index = batch_id * batchsize
		end_index = min((batch_id + 1) * batchsize, data_size)
		yield x[start_index:end_index], y[start_index:end_index]


# -------------------------- dataset -------------------------- #

def load_hole_dataset(sbj_idx, ds_path, shuffle=True, normalization=None, split=None,
					  one_hot=False):  # TODO: has not been tested
	ds = np.load(ds_path, allow_pickle=True)
	x = ds['x']
	y = ds['y']
	del ds

	x = np.concatenate(x[sbj_idx], axis=0)
	x = np.expand_dims(x, axis=1)
	y = np.concatenate(y[sbj_idx], axis=0)[:, -1] // 45
	if one_hot:
		y = one_hot_encoder(y)
	if normalization is not None:
		for i in range(len(x)):
			x[i] = standard_normalization_wise(x[i], normalization)
	if shuffle:
		x, y = shuffle_data([x, y])
	if split is not None:
		split_idx = int(len(y) * split)
		return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]
	return x, y


def load_CYC_dataset(sbj_idx, ds_path, shuffle=True, normalization=None, split=None, label_preprocess=None,
					 label_smooth_para=None):  # TODO: has not been tested
	with open(ds_path, "rb") as fo:
		ds = pickle.load(fo)
	x = ds['x']
	y = ds['y']
	del ds

	x = np.concatenate(x[sbj_idx], axis=0)
	y = np.concatenate(y[sbj_idx], axis=0)
	y = np.array(y[:, -2], dtype=np.int) // 45

	if label_preprocess is None:
		pass
	elif label_preprocess == 'one_hot':
		y = one_hot_encoder(y)
	elif label_preprocess == 'mean_smooth':
		y = one_hot_encoder(y)
		y = smooth_labels_mean(y, factor=label_smooth_para)
	elif label_preprocess == 'gauss_smooth':
		y = one_hot_encoder(y)
		y = smooth_labels_gauss(y, sigma=label_smooth_para)
	else:
		raise ValueError('label_preprocess is not set right!')

	if normalization is not None:
		for i in range(len(x)):
			x[i] = standard_normalization_wise(x[i], normalization)
	if shuffle:
		x, y = shuffle_data([x, y])
	if split is not None:
		split_idx = int(len(y) * split)
		return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]
	return x, y


# -------------------------- label -------------------------- #

def smooth_labels_mean(labels, factor):  # TODO: has not been tested
	labels = np.array(labels)
	labels *= factor
	delta = (1 - factor) / (labels.shape[1] - 1)
	for label in labels:
		label[np.isclose(label, 0)] += delta
	return labels


def smooth_labels_gauss(labels, sigma):  # TODO: has not been tested
	labels = np.array(labels)
	num_cls = labels.shape[-1]
	gauss_template = np.roll([np.exp(- ((i - 0) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)
							  for i in range(num_cls // 2 - num_cls + 1, num_cls // 2 + 1)],
							 shift=num_cls // 2 - num_cls + 1, axis=0)
	print('gauss_template: ', np.around(gauss_template, 3))
	for i, label in enumerate(labels):
		labels[i] = np.roll(gauss_template, shift=np.argmax(label), axis=0)

	return labels


def one_hot_encoder(y, num_classes=None, dtype='float32'):  # TODO: has not been tested
	"""  copied from  tf.keras.utils.to_categorical"""
	y = np.array(y, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype=dtype)
	categorical[np.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = np.reshape(categorical, output_shape)
	return categorical


# -------------------------- utils -------------------------- #

def statistic_label_proportion(train_y, val_y, test_y, do_print=True):  # TODO: has not been tested
	train_y, val_y, test_y, = np.array(train_y), np.array(val_y), np.array(test_y),
	if train_y.ndim > 1:
		train_y = np.argmax(train_y, axis=-1)
	if val_y.ndim > 1:
		val_y = np.argmax(val_y, axis=-1)
	if test_y.ndim > 1:
		test_y = np.argmax(test_y, axis=-1)
	total_y = np.concatenate((train_y, val_y, test_y), axis=0)
	train, val, test, total = Counter(train_y), Counter(val_y), Counter(test_y), Counter(total_y)

	keys = sorted(np.unique(total_y))
	func = lambda key, dict: dict[key] if key in dict.keys() else 0
	sorted_train = [func(key, train) for key in keys]
	sorted_val = [func(key, val) for key in keys]
	sorted_test = [func(key, test) for key in keys]
	sorted_total = [func(key, total) for key in keys]
	if do_print:
		print('\n', '-' * 20, 'Statistical info of dataset labels', '-' * 20)
		print('label: ', keys, )
		print('train: ', sorted_train, )
		print('val:   ', sorted_val, )
		print('test:  ', sorted_test, )
		print('total: ', sorted_total, )

	return {
		'train': sorted_train,
		'val'  : sorted_val,
		'test' : sorted_test,
		'total': sorted_total,
	}


if __name__ == '__main__':
	print('Hello World!')

	print('Brand-new World!')
