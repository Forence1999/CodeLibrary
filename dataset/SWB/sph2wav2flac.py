# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 26/3/2023

from pathlib import Path
from analysis.CodeLibrary.parallel import multiprocess_Pool_async
from analysis.CodeLibrary.audio import check_wav_sample_rate, wav2flac, resample_audio, snip_audio
from analysis.CodeLibrary.path import collect_pattern, convert_extension, convert_rootpath
import subprocess
import warnings

def sph2wav(command):
	'''
	convert sph to wav
	:param command:
	:return:
	'''
	
	global raw_dsroot, wav8k_dsroot
	
	command = command.strip()
	if command == '':
		return
	
	# resolve command
	command = command.rstrip(" |").strip()
	basename, *sph2pipe = command.split()
	
	# generate path
	sph_path = Path(sph2pipe[-1])
	wav_path = (wav8k_dsroot / sph_path.relative_to(raw_dsroot)).parent / (basename + '.wav')
	wav_path.parent.mkdir(parents=True, exist_ok=True)
	
	if wav_path.exists():
		warnings.warn(f"{wav_path} already exists, and will be overwritten")
	
	wav_path.parent.mkdir(parents=True, exist_ok=True)
	return subprocess.run(sph2pipe + [str(wav_path)], shell=False)


def resample_wav(wav8k_path):
	'''
	resample wav from 8k to 16k
	:param wav8k_path:
	:return:
	'''
	global wav8k_dsroot, wav16k_dsroot
	
	wav8k_path = Path(wav8k_path)
	wav16k_path = wav16k_dsroot / wav8k_path.relative_to(wav8k_dsroot)
	
	if check_wav_sample_rate(wav8k_path, 16000):
		return
	else:
		# resample
		wav16k_path.parent.mkdir(parents=True, exist_ok=True)
		resample_audio(wav8k_path, wav16k_path, tgt_sr=16000)


def snip_wav(wav16k_path, segments):
	'''
	based on wav16k, find all the corresponding segments, snip and save the wavs
	:param wav16k_path:
	:param segments:
			e.g.
				sw02001-A_012636-013746 sw02001-A 126.36 137.46
	:return:
	'''
	
	global wav16k_dsroot, snipped_wav16k_dsroot
	
	wav16k_path = Path(wav16k_path)
	basename = wav16k_path.stem
	
	segments = filter(lambda x: x.split()[1] == basename, segments)
	
	for line in segments:
		seg_basename, _, start, end = line.split()
		start, end = int(float(start) * 16000), int(float(end) * 16000)
		
		seg_path = (snipped_wav16k_dsroot / wav16k_path.relative_to(wav16k_dsroot)).parent / \
				   basename / (seg_basename + '.wav')
		seg_path.parent.mkdir(parents=True, exist_ok=True)
		snip_audio(wav16k_path, dst_path=seg_path, start=start, end=end, )


raw_dsroot = Path('/data2/swang/asr/swb/raw')
wav8k_dsroot = Path('/data2/swang/asr/swb/wav_8k')
wav16k_dsroot = Path('/data2/swang/asr/swb/wav_16k')
snipped_wav16k_dsroot = Path('/data2/swang/asr/swb/snipped_wav_16k')
flac_dsroot = Path('/data2/swang/asr/swb/flac')

if __name__ == '__main__':
	# convert sph to wav
	command_ls = Path("./train/wav.scp").read_text().splitlines()
	sph_arguments = [
		{
			"args": (i,)
		}
		for i in command_ls]
	multiprocess_Pool_async(sph2wav, sph_arguments, num_workers=128)
	
	# resample wav from 8k to 16k
	wav_ls = collect_pattern(wav8k_dsroot, '*.wav')
	wav_arguments = [
		{
			"args": (i,)
		}
		for i in wav_ls]
	multiprocess_Pool_async(resample_wav, wav_arguments, num_workers=128)

	# assure all wav files are 16k
	wav_ls = collect_pattern(wav16k_dsroot, pattern='*.wav')
	wavsr_arguments = [
		{
			"args": (i, 16000)
		}
		for i in wav_ls]
	multiprocess_Pool_async(check_wav_sample_rate, wavsr_arguments, num_workers=128)
	
	
	# snip wav16k
	wav16k_ls = collect_pattern(wav16k_dsroot, '*.wav')
	segments = Path('./train/segments').read_text().splitlines()
	segments_arguments = [
		{
			"args": (i, segments)
		}
		for i in wav16k_ls]
	multiprocess_Pool_async(snip_wav, segments_arguments, num_workers=128)
	
	# convert wav to flac
	wav_ls = collect_pattern(snipped_wav16k_dsroot, pattern='*.wav')
	flac_ls = convert_extension(convert_rootpath(wav_ls, snipped_wav16k_dsroot, flac_dsroot), '.flac')
	flac_arguments = [
		{
			"args": i,
			"kwds": {
				"verbose": 1
			}
		}
		for i in zip(wav_ls, flac_ls)]
	multiprocess_Pool_async(wav2flac, flac_arguments, num_workers=128)
