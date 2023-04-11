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
from multiprocessing import Pool


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

    # nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # fp = open("test.txt")
    # data = fp.read()

    # # pattern = r'\.|;|\[|\]|<|>|\?|"|\{|\}|\~|!|\(|\)'
    # pattern = r'\.'
    # result_list = re.split(pattern, line)

    lines = src_path.read_text().splitlines()

    result = []
    for line in lines:
        line = line.strip()
        sentences = tokenizer.tokenize(line)
        for sentence in sentences:
            sentence = re.sub(r"[^a-zA-Z']", " ", sentence)  # only keep letters and apostrophes
            sentence = re.sub(r" +", ' ', sentence).strip().upper()
            if sentence != '':
                result.append(sentence)

    dst_path.write_text('\n'.join(result))


def space2verticalbar(src_path: Path, dst_path: Path):
    print(f'Processing {src_path} ...')

    lines = src_path.read_text().splitlines()

    result = []
    for line in lines:
        line = line.strip().strip("'").strip()
        line = re.sub(r" +", '|', line) + '|'
        result.append(line)

    dst_path.write_text('\n'.join(result))


def split_dataset(src_path: Path, dst_path: Path, ratios):
    src_paths = list(src_path.rglob('*.txt'))
    random.shuffle(src_paths)

    names = ['train', 'valid', 'test']
    start_idx = 0
    for i, ratio in enumerate(ratios):
        name = names[i]
        end_idx = len(src_paths) if i == len(ratios) - 1 else int(len(src_paths) * ratio) + start_idx

        for f in src_paths[start_idx:end_idx]:
            dst_paths = dst_path / name / f.relative_to(src_path)
            dst_paths.parent.mkdir(parents=True, exist_ok=True)
            dst_paths.symlink_to(f)

        start_idx = end_idx


def gen_raw_text(src_dir: Path, dst_dir: Path):
    name = src_dir.name + '.raw'
    dst_path = dst_dir / name

    result = [f.read_text() for f in src_dir.rglob('*.txt')]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text('\n'.join(result))


if __name__ == '__main__':
    num_workers = 128

    src_ds_root = Path('/data/swang/asr/LibriSpeech/lm/librispeech-lm-corpus/corpus')
    dst_para_ds_root = Path('/data/swang/librispeech-lm-corpus/paragraph')
    dst_filtered_ds_root = Path('/data/swang/librispeech-lm-corpus/filtered')
    dst_sentence_ds_root = Path('/data/swang/librispeech-lm-corpus/sentence')
    dst_verticalbar_ds_root = Path('/data/swang/librispeech-lm-corpus/verticalbar')
    dst_split_ds_root = Path('/data/swang/librispeech-lm-corpus/split')
    dst_raw_ds_root = Path('/data/swang/librispeech-lm-corpus/raw')

    # pool = Pool(processes=num_workers)
    # for src_path in src_ds_root.rglob('*.txt'):
    #     dst_path = dst_para_ds_root / src_path.relative_to(src_ds_root)
    #     dst_path.parent.mkdir(parents=True, exist_ok=True)
    #     pool.apply_async(sentence2paragraph, args=(src_path, dst_path))
    # pool.close()
    # pool.join()
    #
    # # pool = Pool(processes=num_workers)
    # for src_path in dst_para_ds_root.rglob('*.txt'):
    #     dst_path = dst_filtered_ds_root / src_path.relative_to(dst_para_ds_root)
    #     dst_path.parent.mkdir(parents=True, exist_ok=True)
    #     pool.apply_async(filter_short_paragraph, args=(src_path, dst_path))
    # pool.close()
    # pool.join()
    #
    # # pool = Pool(processes=num_workers)
    # for src_path in dst_filtered_ds_root.rglob('*.txt'):
    #     dst_path = dst_sentence_ds_root / src_path.relative_to(dst_filtered_ds_root)
    #     dst_path.parent.mkdir(parents=True, exist_ok=True)
    #     pool.apply_async(paragraph2sentence, args=(src_path, dst_path))
    # pool.close()
    # pool.join()

    pool = Pool(processes=num_workers)
    for src_path in dst_sentence_ds_root.rglob('*.txt'):
        dst_path = dst_verticalbar_ds_root / src_path.relative_to(dst_sentence_ds_root)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pool.apply_async(space2verticalbar, args=(src_path, dst_path))
    pool.close()
    pool.join()

    split_ratio = [0.6, 0.2, 0.2]
    split_dataset(dst_verticalbar_ds_root, dst_split_ds_root, split_ratio)

    for src_dir in dst_split_ds_root.glob('*'):
        gen_raw_text(src_dir, dst_raw_ds_root)

    print('Hello World!')
