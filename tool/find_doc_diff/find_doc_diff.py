# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 28/12/2022

import os, sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
import difflib
from functools import reduce
import re
from collections import Counter


def Compare(doc_1, doc_2, ofile):
    
    # diff = difflib.HtmlDiff()  # 创建一个diff对象
    # # result = diff.make_table(doc_1, doc_2)  # 得出比较结果
    # # result = diff.make_file(doc_1, doc_2, context=True, numlines=0)  # 得出比较结果
    # result = diff.make_table(doc_1, doc_2, context=True, numlines=0)  # 得出比较结果
    # fd_diff = open(ofile, "w")
    # fd_diff.write(result)
    # fd_diff.close()
    
    diffs = difflib._mdiff(doc_1, doc_2, context=0, charjunk=difflib.IS_CHARACTER_JUNK, linejunk=None)
    result = []
    
    for i, j, tag in diffs:
        if i is None or j is None:
            continue
        i, j = i[1], j[1]
        i, j = re.sub(r'[\x00\x01^+\-]', '', i), re.sub(r'[\x00\x01^+\-]', '', j)
        if i == '\n':
            result[-1][1] = result[-1][1] + ' ' + j.replace('\n', ' ')
        elif j == '\n':
            result[-1][0] = result[-1][0] + ' ' + i.replace('\n', ' ')
        else:
            result.append([i, j.replace('\n', ' ')])
    ref = [i[0] for i in result]
    ref = dict(Counter(ref).most_common())
    for k, v in ref.items():
        ref[k] = [str(v)]
    for i in result:
        if i[0] in ref:
            ref[i[0]].append(i[1])
    
    for k, v in ref.items():
        print('-' * 100)
        print(k, '\n', '\n'.join(v))


def preprocess(doc):
    def concat(ls, x):
        return ls + x.split(' ')
    
    doc = Path(doc).read_text().splitlines()
    
    return reduce(concat, [[]] + doc)


if __name__ == '__main__':
    # doc_1 = './ref.word-checkpoint_best.pt-dev_other.txt'
    # doc_2 = './hypo.word-checkpoint_best.pt-dev_other.txt'
    doc_1 = './ref.word-checkpoint_best.pt-test_other.txt'
    doc_2 = './hypo.word-checkpoint_best.pt-test_other.txt'
    # doc_1 = './wo_dtw.hypo.word-checkpoint_best.pt-test_other.txt'
    # doc_2 = './w_dtw.hypo.word-checkpoint_best.pt-test_other.txt'
    # doc_1 = './wo_dtw.hypo.word-checkpoint_best.pt-dev_other.txt'
    # doc_2 = './w_dtw.hypo.word-checkpoint_best.pt-dev_other.txt'
    # ofile = './wow_sdtw.word-checkpoint_best.pt-dev_other.html'
    
    ofile = './temp.html'
    
    doc_1 = preprocess(doc_1)
    doc_2 = preprocess(doc_2)
    
    Compare(doc_1, doc_2, ofile)
