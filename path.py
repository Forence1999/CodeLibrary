# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 28/3/2023

import os, sys
import time
import random
import warnings
import numpy as np
from pathlib import Path

def convert_rootpath(file_ls, src_root, dst_root):
	return [dst_root / Path(i).relative_to(src_root) for i in file_ls]


def convert_extension(file_ls, extension):
	return [i.with_suffix(extension) for i in file_ls]


def collect_pattern(dsroot, pattern):
	return [i for i in dsroot.rglob(pattern)]

