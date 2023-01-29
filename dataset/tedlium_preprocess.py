# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 29/1/2023


from pathlib import Path
import re


def filter_audio_with_unknown_tag(audio_path: Path, dest_dir: Path):
    '''
    Filter the audios with unknown tag
    audio_path: Path to the audio tsv file
    dest_dir: Path to the destination directory
    '''
    audio_lines = audio_path.read_text().splitlines()
    root_path = audio_lines[0].strip()
    audio_lines = audio_lines[1:]
    
    wrd_path = audio_path.with_suffix('.wrd')
    wrd_lines = wrd_path.read_text().splitlines()
    for i in range(len(wrd_lines) - 1, -1, -1):
        if ('<UNK>' in wrd_lines[i]) or ('<unk>' in wrd_lines[i]):
            audio_lines.pop(i)
            wrd_lines.pop(i)
    
    # normalize and save the path
    dest_csv = dest_dir / audio_path.name
    with open(dest_csv, "w") as tsv_out, \
            open(dest_csv.with_suffix(".ltr"), "w") as ltr_out, \
            open(dest_csv.with_suffix(".wrd"), "w") as wrd_out:
        print(root_path, file=tsv_out)
    
        for csv_line, wrd_line in zip(audio_lines, wrd_lines):
            # preprocess the wrd_line
            wrd_line = wrd_line.strip()
            wrd_line = re.sub(r"[^[a-zA-Z]']", '', wrd_line)  # only keep letters and apostrophes
            wrd_line = re.sub(r" +", ' ', wrd_line).replace(" '", "'").upper()
            
            char_line = " ".join(list(wrd_line.replace(" ", "|"))) + " |"
            
            print(csv_line, file=tsv_out)
            print(wrd_line, file=wrd_out)
            print(char_line, file=ltr_out)


if __name__ == "__main__":
    print('Hello World!')
    
    splits = ['train', 'dev', 'test']
    
    path_pattern = '/data2/swang/asr/TEDLIUM_release-3/legacy_cropped/unfiltered/{split}.tsv'
    dest_dir = Path('/data2/swang/asr/TEDLIUM_release-3/legacy_cropped/filtered')
    for split in splits:
        print(f'Processing {split}...')
        audio_path = Path(path_pattern.format(split=split))
        filter_audio_with_unknown_tag(audio_path, dest_dir=dest_dir)
