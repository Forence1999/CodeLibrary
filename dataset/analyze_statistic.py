# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 28/1/2023

import os, sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from analysis.CodeLibrary.visualization import plot_hist


def calcu_total_duration(path, filter_unknown=False, return_seconds=False):
    '''
    Calculate the total duration of the dataset.
    unit: hour
    path: the path of the tsv file
    filter_unknown: whether to filter the audios with unknown tag
    return_seconds: whether to return the second of the audios
    '''
    
    audio_path = Path(path)
    audio_lines = audio_path.read_text().splitlines()[1:]
    
    if filter_unknown:
        wrd_path = audio_path.with_suffix('.wrd')
        wrd_lines = wrd_path.read_text().splitlines()
        for i in range(len(wrd_lines) - 1, -1, -1):
            if '<UNK>' in wrd_lines[i]:
                audio_lines.pop(i)
                wrd_lines.pop(i)
    
    frames = [int(line.split('\t')[1]) for line in audio_lines]
    total_duration = sum(frames) / 16000 / 3600
    
    if return_seconds:
        return total_duration, [frame / 16000 for frame in frames]
    else:
        return total_duration


def plot_duration_distribution(path=None, name=None, durations=None):
    '''
    Plot the distribution of the duration of the dataset.
    path: the path of the tsv file
    name: the name of the dataset
    durations: the list of the duration of the dataset
    path has higher priority than durations
    '''
    assert path is not None or durations is not None
    
    if path is not None:
        lines = Path(path).read_text().splitlines()
        with open(path, 'r') as f:
            f.readlines()
        frames = [int(line.split('\t')[1]) for line in lines[1:]]
        durations = [frame / 16000 for frame in frames]
    
    title = 'Duration Distribution of the {}'.format(name)
    names = [name, ]
    data = [durations, ]
    color = ['b', ]
    plot_hist(data=list(zip(names, data, color)), title=title, img_path='./{}.png'.format(name))


if __name__ == '__main__':
    print('Hello World!')
    # splits = ['wo_preprocess/train', 'wo_preprocess/dev', 'wo_preprocess/test']
    # splits = ['AZ_apostrophe/train', 'AZ_apostrophe/dev', 'AZ_apostrophe/test']
    splits = ['10m/train', '1h/train', '10h/train', '100h/train']
    path_pattern = '/home/swang/project/smartspeaker/asr/fairseq/examples/wav2vec/dataset/TED-LIUM3/{split}.tsv'
    for split in splits:
        path = path_pattern.format(split=split)
        dur_h, seconds = calcu_total_duration(path, filter_unknown=False, return_seconds=True)
        print('The total duration of the {} set (hour): {:.2f}'.format(split, dur_h))
        plot_duration_distribution(durations=seconds, name=split)
