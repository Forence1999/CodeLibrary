# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 29/1/2023
import warnings
from pathlib import Path
import random


def sample_audio(tsv_path: Path, wrd_path: Path, ltr_path: Path, dst_dir: Path, duration: int):
    '''
    Sample audio counting the specified duration
    :param tsv_path: the path of the audio tsv file
    :param wrd_path: the path of the wrd file
    :param ltr_path: the path of the ltr file
    :param dst_dir: the path of the destination directory
    :param duration: the duration of the audio (seconds)
    '''
    
    dst_tsv_path = dst_dir / tsv_path.name
    dst_wrd_path = dst_dir / wrd_path.name
    dst_ltr_path = dst_dir / ltr_path.name
    
    tsv_lines = tsv_path.read_text().splitlines()
    wrd_lines = wrd_path.read_text().splitlines()
    ltr_lines = ltr_path.read_text().splitlines()
    root_path = tsv_lines.pop(0).strip()
    
    # sample randomly
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_tsv_path, 'w') as tsv_out, open(dst_wrd_path, 'w') as wrd_out, open(dst_ltr_path, 'w') as ltr_out:
        print(root_path, file=tsv_out)
        
        selected_duration = 0
        index_ls = list(range(len(tsv_lines)))
        while selected_duration < duration:
            if len(index_ls) == 0:
                warnings.warn(f'No audio left for sampling. The duration of {dst_tsv_path} is {selected_duration:.2f} \
				seconds, which might be less than the specified duration {duration} seconds.')
                break
            idx = random.choice(index_ls)
            index_ls.remove(idx)
            print(tsv_lines[idx], file=tsv_out)
            print(wrd_lines[idx], file=wrd_out)
            print(ltr_lines[idx], file=ltr_out)
            selected_duration += int(tsv_lines[idx].split('\t')[-1]) / 16000.
        
        print(f'{dst_tsv_path}: {selected_duration:.2f} seconds')


if __name__ == "__main__":
    
    duration_dict = {
        '10m': 1. * 3600 * 1 / 6,
        '1h' : 1. * 3600 * 1,
        '10h': 1. * 3600 * 10,
    }
    
    file_dir = Path(
        '/home/swang/project/smartspeaker/asr/fairseq/analysis/CodeLibrary/dataset/TED-LIUM3/processed/noisy')
    tsv_path = file_dir / 'train.orig.tsv'
    ltr_path = tsv_path.with_suffix('.ltr')
    wrd_path = tsv_path.with_suffix('.wrd')
    
    for name in duration_dict.keys():
        duration = duration_dict[name]
        dst_dir = file_dir / name
        
        sample_audio(tsv_path, wrd_path, ltr_path, dst_dir=dst_dir, duration=duration)
