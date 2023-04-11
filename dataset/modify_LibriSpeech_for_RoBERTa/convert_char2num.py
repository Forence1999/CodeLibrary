# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 7/2/2023

import os, sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
import json


def char2num(src_path, dest_path):
    dict = src_path.read_text().splitlines()
    
    res = []
    for i in dict:
        if i == '':
            continue
        token, freq = i.strip().split()
        token = str(ord(token.strip()))
        res.append(token + ' ' + freq)
    
    dest_path.write_text('\n'.join(res))


def gen_encoder(dict_path, dest_path):
    dict = dict_path.read_text().splitlines()
    tokens = [i.strip().split()[0] for i in dict if i.strip() != '']
    
    encoder = {i: ord(i) for i in tokens}
    
    dest_path.write_text(json.dumps(encoder))


if __name__ == '__main__':
    dict_path = Path('dict.txt')
    dest_dict_path = Path('dict_num.txt')
    dest_encoder_path = Path('encoder_num.json')
    
    # char2num(dict_path, dest_dict_path)
    gen_encoder(dict_path, dest_encoder_path)
    print('Hello World!')
