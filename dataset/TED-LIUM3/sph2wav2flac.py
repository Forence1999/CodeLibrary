# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 26/3/2023
import warnings
from pathlib import Path
from analysis.CodeLibrary.parallel import multiprocess_Pool_async
from analysis.CodeLibrary.audio import check_wav_sample_rate, wav2flac, resample_audio, snip_audio
from analysis.CodeLibrary.path import collect_pattern, convert_extension, convert_rootpath
import subprocess


def sph2wav(command):
    '''
    convert sph to wav
    :param command:
    :return:
    '''
    
    global raw_dsroot, wav_dsroot, sph2pipe
    
    command = command.strip()
    if command == '':
        return
    
    # resolve command
    command = command.rstrip(" |").strip()
    basename, *sph2pipe_ls = command.split()
    sph2pipe_ls[0] = sph2pipe
    
    # generate path
    sph_path = Path(sph2pipe_ls[-1]).resolve()
    wav_path = (wav_dsroot / sph_path.relative_to(raw_dsroot)).parent / (basename + '.wav')
    
    if wav_path.exists():
        warnings.warn(f"{wav_path} already exists, and will be overwritten")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(sph2pipe_ls + [str(wav_path)], shell=False)


def snip_wav(wav_path, segments):
    '''
    based on wav, find all the corresponding segments, snip and save the wavs
    :param wav_path:
    :param segments:
            e.g.
                AimeeMullins_2009P-0001782-0002881 AimeeMullins_2009P 0017.82 0028.81
    :return:
    '''
    
    global wav_dsroot, snipped_wav_dsroot
    
    wav_path = Path(wav_path)
    basename = wav_path.stem
    
    segments = filter(lambda x: x.split()[1] == basename, segments)
    
    for line in segments:
        seg_basename, _, start, end = line.split()
        start, end = int(float(start) * 16000), int(float(end) * 16000)
        
        seg_path = (snipped_wav_dsroot / wav_path.relative_to(wav_dsroot)).parent / \
                   basename / (seg_basename + '.wav')
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        snip_audio(wav_path, dst_path=seg_path, start=start, end=end, )


sph2pipe = "/home/swang/software/kaldi/tools/sph2pipe_v2.5/sph2pipe"
raw_dsroot = Path('/data2/swang/asr/TEDLIUM_release-3/raw')
wav_dsroot = Path('/data2/swang/asr/TEDLIUM_release-3/wav')
snipped_wav_dsroot = Path('/data2/swang/asr/TEDLIUM_release-3/snipped_wav')
flac_dsroot = Path('/data2/swang/asr/TEDLIUM_release-3/flac')

if __name__ == '__main__':
    
    dataset = ['test', 'dev', 'train']
    
    for dataset_split in dataset:
        dataset_split += ".orig"
        print(f"processing {dataset_split}")
        
        # convert sph to wav
        command_ls = Path(f"./{dataset_split}/wav.scp").read_text().splitlines()
        sph_arguments = [
            {
                "args": (i,)
            }
            for i in command_ls]
        # sph2wav(command_ls[0])
        multiprocess_Pool_async(sph2wav, sph_arguments, num_workers=128)
        
        # assure all wav files are 16k
        wav_ls = collect_pattern(wav_dsroot, pattern='*.wav')
        wavsr_arguments = [
            {
                "args": (i, 16000)
            }
            for i in wav_ls]
        multiprocess_Pool_async(check_wav_sample_rate, wavsr_arguments, num_workers=128)
        
        # snip wav
        wav_ls = collect_pattern(wav_dsroot, '*.wav')
        segments = Path(f'./{dataset_split}/segments').read_text().splitlines()
        segments_arguments = [
            {
                "args": (i, segments)
            }
            for i in wav_ls]
        # snip_wav(wav_ls[0], segments)
        multiprocess_Pool_async(snip_wav, segments_arguments, num_workers=128)
        
        # convert wav to flac
        wav_ls = collect_pattern(snipped_wav_dsroot, pattern='*.wav')
        flac_ls = convert_extension(convert_rootpath(wav_ls, snipped_wav_dsroot, flac_dsroot), '.flac')
        flac_arguments = [
            {
                "args": i,
                "kwds": {
                    "verbose": 1
                }
            }
            for i in zip(wav_ls, flac_ls)]
        multiprocess_Pool_async(wav2flac, flac_arguments, num_workers=128)
