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
import re


def sentence2paragraph(src_path: Path, dst_path: Path):
    print(f'Processing {src_path} ...')
    
    lines = src_path.read_text().splitlines()
    
    result = []
    buffer = []
    for line in lines:
        line = line.strip()
        if line == '':
            if len(buffer) > 0:
                paragraph = ' '.join(buffer)
                paragraph = re.sub(r" +", " ", paragraph)
                result.append(paragraph)
                buffer = []
            result.append('')
        else:
            buffer.append(line)
    if len(buffer) > 0:
        paragraph = ' '.join(buffer)
        paragraph = re.sub(r" +", " ", paragraph)
        result.append(paragraph)
    
    dst_path.write_text('\n'.join(result))


def filter_short_paragraph(src_path: Path, dst_path: Path):
    print(f'Processing {src_path} ...')
    
    lines = src_path.read_text().splitlines()
    
    result = [line for line in lines if len(line.split()) > 20]
    
    dst_path.write_text('\n'.join(result))


def paragraph2sentence(src_path: Path, dst_path: Path):
    print(f'Processing {src_path} ...')

    import nltk.data

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open("test.txt")
    data = fp.read()
    sentences=tokenizer.tokenize(data)
    # pattern = r'\.|;|\[|\]|<|>|\?|"|\{|\}|\~|!|\(|\)'
    pattern = r'\.'
    
    lines = src_path.read_text().splitlines()
    
    result = []
    for line in lines:
        line = line.strip()
        result_list = re.split(pattern, line)

        result.extend(result_list)
    
    dst_path.write_text('\n'.join(result))


if __name__ == '__main__':
    
    src_ds_root = Path('/data/swang/asr/LibriSpeech/lm/librispeech-lm-corpus/corpus')
    dst_para_ds_root = Path('/data/swang/librispeech-lm-corpus/paragraph')
    dst_filtered_ds_root = Path('/data/swang/librispeech-lm-corpus/filtered')
    dst_sentence_ds_root = Path('/data/swang/librispeech-lm-corpus/sentence')
    
    # for src_path in src_ds_root.rglob('*.txt'):
    #     dst_path = dst_para_ds_root / src_path.relative_to(src_ds_root)
    #     dst_path.parent.mkdir(parents=True, exist_ok=True)
    #     sentence2paragraph(src_path, dst_path)
    #
    #     break
    #
    # for src_path in dst_para_ds_root.rglob('*.txt'):
    #     dst_path = dst_filtered_ds_root / src_path.relative_to(dst_para_ds_root)
    #     dst_path.parent.mkdir(parents=True, exist_ok=True)
    #     filter_short_paragraph(src_path, dst_path)
    #
    #     break
    
    for src_path in dst_filtered_ds_root.rglob('*.txt'):
        dst_path = dst_sentence_ds_root / src_path.relative_to(dst_filtered_ds_root)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        paragraph2sentence(src_path, dst_path)
        
        break


    print('Hello World!')
