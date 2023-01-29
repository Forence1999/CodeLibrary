# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 28/1/2023

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
"""

import argparse
import os
from pathlib import Path
import re


def main(args):
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.tsv, "r") as tsv, \
            open(os.path.join(args.output_dir, args.output_name + ".ltr"), "w") as ltr_out, \
            open(os.path.join(args.output_dir, args.output_name + ".wrd"), "w") as wrd_out:
        root = Path(next(tsv).strip())
        for line in tsv:
            line = line.strip().split("\t")[0]
            trans_path = (root / line).with_suffix('.txt')
            with open(trans_path, "r") as t:
                trans = next(t).strip()
            # trans = re.sub(r"[^[a-zA-Z]']", '', trans)  # only keep letters and apostrophes
            trans = re.sub(r" +", ' ', trans).replace(" '", "'").upper()
            print(trans, file=wrd_out)
            print(" ".join(list(trans.replace(" ", "|"))) + " |", file=ltr_out, )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv",
                        default="/data2/swang/asr/TEDLIUM_release-3/legacy_cropped/unfiltered/test.tsv")
    parser.add_argument("--output-dir",
                        default="/data2/swang/asr/TEDLIUM_release-3/legacy_cropped/unfiltered")
    parser.add_argument("--output-name",
                        default="test")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
