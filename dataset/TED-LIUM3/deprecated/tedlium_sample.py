# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 29/1/2023


from pathlib import Path
import random


def sample_audio(audio_path: Path, dest_dir: Path, duration: int):
    '''
    Filter the audios with unknown tag
    :param audio_path: the path of the audio csv file
    :param dest_dir: the path of the destination directory
    :param duration: the duration of the audio (seconds)
    '''
    src_csv = audio_path.read_text().splitlines()
    root_path = src_csv[0].strip()
    src_csv = src_csv[1:]
    wrd_path = audio_path.with_suffix('.wrd')
    src_wrd = wrd_path.read_text().splitlines()
    ltr_path = audio_path.with_suffix('.ltr')
    src_ltr = ltr_path.read_text().splitlines()
    
    # sample randomly
    selected_duration = 0
    selected_data = []
    while selected_duration < duration:
        length = len(src_csv)
        assert length > 0, f'No audio left in {audio_path}'
        idx = random.choice(range(length))
        data = (src_csv.pop(idx), src_wrd.pop(idx), src_ltr.pop(idx))
        selected_data.append(data)
        
        selected_duration += int(data[0].split('\t')[-1]) / 16000
    
    # sort
    selected_data = sorted(selected_data, key=lambda x: x[0].split('\t')[0])
    frames = [int(data[0].split('\t')[-1]) for data in selected_data]
    selected_duration = sum(frames) / 16000 / 3600
    print(f'Selected {len(selected_data)} audios, {selected_duration:.2f} hours')
    
    # write
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = dest_dir / audio_path.name
    with open(dest_csv, "w") as tsv_out, \
            open(dest_csv.with_suffix(".ltr"), "w") as ltr_out, \
            open(dest_csv.with_suffix(".wrd"), "w") as wrd_out:
        print(root_path, file=tsv_out)
        for data in selected_data:
            print(data[0], file=tsv_out)
            print(data[1], file=wrd_out)
            print(data[2], file=ltr_out)


if __name__ == "__main__":
    print('Hello World!')
    
    splits = ['train', ]
    
    path_pattern = '/home/swang/project/smartspeaker/asr/fairseq/examples/wav2vec/dataset/TED-LIUM3/AZ_apostrophe/{split}.tsv'
    dest_dir = Path('/home/swang/project/smartspeaker/asr/fairseq/examples/wav2vec/dataset/TED-LIUM3/10m')
    duration = 1. * 3600 * 1 / 6  # seconds
    for split in splits:
        print(f'Processing {split}...')
        audio_path = Path(path_pattern.format(split=split))
        sample_audio(audio_path, dest_dir=dest_dir, duration=duration)
