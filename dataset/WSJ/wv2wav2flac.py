# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 26/3/2023

from pathlib import Path
from analysis.CodeLibrary.parallel import multiprocess_Pool_async
from analysis.CodeLibrary.audio import check_wav_sample_rate, wav2flac
from analysis.CodeLibrary.path import collect_pattern, convert_extension, convert_rootpath
import subprocess
import warnings


def collect_files_from_all_wav_scp(wav_scp_dir):
	file_ls = []
	for wav_scp in wav_scp_dir.glob('*_wav.scp'):
		lines = wav_scp.read_text().split('\n')
		file_ls.extend(lines)
	return file_ls


def wv2wav(wv_path):
	global raw_dsroot, wav_dsroot, sph2pipe
	
	wav_path = (wav_dsroot / Path(wv_path).relative_to(raw_dsroot)).with_suffix('.wav')
	
	if wav_path.exists():
		warnings.warn(f"{wav_path} already exists, and will be overwritten")
	
	wav_path.parent.mkdir(parents=True, exist_ok=True)
	return subprocess.run([sph2pipe, "-f", "wav", str(wv_path), str(wav_file)], shell=False)


raw_dsroot = Path('/data2/swang/asr/wsj/raw')
wav_dsroot = Path('/data2/swang/asr/wsj/wav')
flac_dsroot = Path('/data2/swang/asr/wsj/flac')
# wav_scp_dir = Path('/data2/swang/asr/wsj/kaldi_data/data')
sph2pipe = "/home/swang/software/kaldi/tools/sph2pipe_v2.5/sph2pipe"

if __name__ == '__main__':
	
	# info_lines = collect_files_from_all_wav_scp(wav_scp_dir)
	
	# convert wv1 to wav
	wv1_ls = collect_pattern(raw_dsroot, pattern='*.wv1')
	wv1_arguments = [
		{
			"args": (i,)
		}
		for i in wv1_ls]
	multiprocess_Pool_async(wv2wav, wv1_arguments, num_workers=128)
	
	# assure all wav files are 16k
	wav_ls = collect_pattern(wav_dsroot, pattern='*.wav')
	wavsr_arguments = [
		{
			"args": (i, 16000)
		}
		for i in wav_ls]
	multiprocess_Pool_async(check_wav_sample_rate, wavsr_arguments, num_workers=128)
	
	# convert wav to flac
	wav_ls = collect_pattern(wav_dsroot, pattern='*.wav')
	flac_ls = convert_extension(convert_rootpath(wav_ls, wav_dsroot, flac_dsroot), '.flac')
	flac_arguments = [
		{
			"args": i,
			"kwds": {
				"verbose": 1
			}
		}
		for i in zip(wav_ls, flac_ls)]
	multiprocess_Pool_async(wav2flac, flac_arguments, num_workers=128)
