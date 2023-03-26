# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 26/3/2023

import os, sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
from analysis.CodeLibrary.parallel import multiprocess_Pool
import subprocess
import warnings


def collect_files_from_all_wav_scp(wav_scp_dir):
	file_ls = []
	for wav_scp in wav_scp_dir.glob('*_wav.scp'):
		lines = wav_scp.read_text().split('\n')
		file_ls.extend(lines)
	return file_ls


def collect_wv1(raw_dsroot):
	
	return [i for i in raw_dsroot.rglob('*.wv1')]


def convert_wav_to_flac(file_ls):
	for file in file_ls:
		file = file.split(' ')[-1]
		file = Path(file)
		if file.suffix == '.wav':
			file_flac = file.with_suffix('.flac')
			if not file_flac.exists():
				os.system(f'ffmpeg -i {file} {file_flac}')
		elif file.suffix == '.flac':
			pass
		else:
			raise ValueError(f'file {file} is not wav or flac')
		print(f'processed {file}')


def wv2wav(wv_path):
	global raw_dsroot, wav_dsroot, sph2pipe
	
	wav_file = (wav_dsroot / Path(wv_path).relative_to(raw_dsroot)).with_suffix('.wav')
	wav_file.parent.mkdir(parents=True, exist_ok=True)
	if not wav_file.exists():
		# print(f'processing {wav_file}')
		return subprocess.run([sph2pipe, "-f", "wav", str(wv_path), str(wav_file)], shell=False)
	else:
		print(wv_path)


# warnings.warn(f'file {wav_file} already exists')


raw_dsroot = Path('/data2/swang/asr/wsj/raw')
wav_dsroot = Path('/data2/swang/asr/wsj/wav')
# wav_scp_dir = Path('/data2/swang/asr/wsj/kaldi_data/data')
sph2pipe = "/home/swang/software/kaldi/tools/sph2pipe_v2.5/sph2pipe"

# if info_line.strip() == '':
# 	return 1
# else:
# 	id, sph2pipe, _, _, wv_path, _ = info_line.split(' ')


if __name__ == '__main__':
	
	# info_lines = collect_files_from_all_wav_scp(wav_scp_dir)
	
	wv1_ls = collect_wv1(raw_dsroot)
	multiprocess_Pool(wv2wav, wv1_ls, num_workers=128)
	
	# convert_wav_to_flac(file_ls)
	
	print('Hello World!')
