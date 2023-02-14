# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022

import os
import random
import re
import sys

import numpy as np


def next_greater_power_of_2(x):
	return 2 ** (int(x) - 1).bit_length()


def next_lower_power_of_2(x):
	return 2 ** ((int(x) - 1).bit_length() - 1)


def set_random_seed(seed=0, fix_np=False, fix_tf=False, fix_torch=False, ):
	''' setting random seed '''
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	if fix_np:
		np.random.seed(seed)
	if fix_tf:
		import tensorflow as tf

		tf.random.set_seed(seed)
	if fix_torch:
		import torch
		import torch.backends.cudnn as cudnn

		torch.manual_seed(seed)
		cudnn.deterministic = True
		print('Warning:', 'You have chosen to seed training. '
						  'This will turn on the CUDNN deterministic setting, '
						  'which can slow down your training considerably! '
						  'You may see unexpected behavior when restarting '
						  'from checkpoints.')


def extract_nb_from_str(str):  # TODO: has not been tested
	pattern = re.compile(r'\d+')
	res = re.findall(pattern, str)
	return list(map(int, res))


# -------------------------- print -------------------------- #

def enable_printing():  # TODO: has not been tested
	sys.stdout = sys.__stdout__
	sys.stderr = sys.__stderr__
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def disable_printing():  # TODO: has not been tested
	sys.stdout = open(os.devnull, 'w')
	sys.stderr = open(os.devnull, 'w')
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
