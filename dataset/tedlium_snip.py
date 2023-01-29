# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 28/1/2023
# This file is modified from https://gitlab.labranet.jamk.fi/data-analysis-and-ai/speech-to-text/-/blob/master/scripts/tedlium-snip.py

# Copyright (C) 2018 JAMK University of Applied Sciences
#
# This software may be modified and distributed under the terms
# of the MIT license. See the LICENSE file for details.

import re
import os
import sys
import argparse
import os.path
import multiprocessing
import subprocess
import functools
from pathlib import Path


class Transcription:
    
    def __init__(self, start_time, end_time, transcription, pronunciation):
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.transcription = transcription
        self.pronunciation = pronunciation
    
    def get_start_time(self):
        return self.start_time
    
    def get_end_time(self):
        return self.end_time
    
    def get_transcription(self):
        return self.transcription
    
    def get_pronunciation(self):
        return self.pronunciation


def get_pronunciation(dictionary, transcription):
    pronunciation = []
    for word in transcription:
        p = dictionary.get(word, None)
        if p:
            pronunciation.append(p)
    return pronunciation


def parse_transcriptions(dictionary, file_name):
    transcriptions = []
    with open(file_name) as f:
        for line in f:
            split = line.split()
            _, _, _, start_time, end_time, _ = split[0:6]
            transcription = split[6:]
            if transcription[0] == 'ignore_time_segment_in_scoring':
                continue
            pronunciation = get_pronunciation(dictionary, transcription)
            t = Transcription(start_time, end_time, transcription, pronunciation)
            transcriptions.append(t)
    return transcriptions


def load_dictionary(dic_file):
    dictionary = {}
    p = re.compile('([\-\w\']+)(\(\d+\))? +([A-Z ]+)$')
    with open(dic_file) as f:
        for line in f:
            result = p.search(line)
            if result:
                word, nbr, pronunciation = result.groups()
                if not nbr:
                    dictionary[word] = pronunciation
            else:
                print(line)
    
    return dictionary


def write_pronunciation(filename, transcription):
    with open(filename, 'w') as f:
        content = '\n'.join(
            [' '.join(transcription.get_transcription())] + transcription.get_pronunciation())
        f.write(content)


def process(dictionary, out_dir, args):
    '''
    snip a single audio file to segments according to its transcription file.
    all the segments and their corresponding transcripts are saved in out_dir.
    file names are formatted as <basename>-<i>.wav and <basename>-<i>.txt.
    '''
    devnull = open(os.devnull, 'w')
    sph_file, stm_file = args
    transcriptions = parse_transcriptions(dictionary, stm_file)
    ffmpeg_command = 'ffmpeg -y -i {in_file} -ss {start_time} -to {stop_time} {out_file}'
    basename = os.path.splitext(os.path.basename(sph_file))[0]
    out_filename = basename + '-{i}.flac'
    os.makedirs(os.path.join(out_dir, basename), exist_ok=True)
    out_path = os.path.join(out_dir, basename, out_filename)
    
    i = 0
    for t in transcriptions:
        cmd = ffmpeg_command.format(
            in_file=sph_file,
            start_time=t.get_start_time(),
            stop_time=t.get_end_time(),
            out_file=out_path.format(i=i))
        subprocess.call(cmd.split(), stdout=devnull, stderr=devnull)
        pronunciation_file = os.path.join(out_dir, basename, basename + '-{i}.txt'.format(i=i))
        write_pronunciation(pronunciation_file, t)
        sys.stdout.write('.')
        sys.stdout.flush()
        i += 1


def main(stm_files, sph_files, dic_file, out_dir, num_processes):
    
    dictionary = load_dictionary(dic_file)
    
    # Check that all the audio files have matching transcription file
    assert len(stm_files) == len(sph_files), \
        'All audio files do not have matching transcription file.'
    stm_files.sort()
    sph_files.sort()
    for i in range(len(stm_files)):
        stm_base = os.path.splitext(os.path.basename(stm_files[i]))[0]
        sph_base = os.path.splitext(os.path.basename(sph_files[i]))[0]
        assert stm_base == sph_base, \
            'Audiofile {} does not have matching transcription'.format(sph_base)
    
    os.makedirs(out_dir, exist_ok=True)
    
    pool = multiprocessing.Pool(num_processes)
    process_fn = functools.partial(process, dictionary, out_dir)
    pool.map(process_fn, [(sph_files[i], stm_files[i]) for i in range(len(sph_files))])
    # process(dictionary, out_dir, (sph_files[0], stm_files[0]))


if __name__ == '__main__':
    ds_root = Path('/data2/swang/asr/TEDLIUM_release-3')
    
    src_dir = ds_root / 'legacy' / 'test'
    print('Processing files in {}'.format(src_dir))
    stm_files = list((src_dir / 'stm').rglob('*.stm'))
    sph_files = list((src_dir / 'sph').rglob('*.sph'))
    dic_file = ds_root / 'TEDLIUM.152k.dic'
    
    out_dir = ds_root / 'legacy_cropped' / 'test'
    
    num_processes = 1
    
    main(stm_files, sph_files, dic_file, out_dir, num_processes)
    
    '''
    Split the dataset over and under n kb files
    Even after the cropping process, some of the audio files are super long. The
    following commands can be used to split the dataset to over/under partitions.
    In the following commands the split is done with 325kb as a limit.
    
    # move too big files to a different folder. 325kb wav file is about 10 secs, with
    # the default mfcc frame size (0,025 sec), the max sequence length can be set to 500.
    mkdir under-325kb
    mkdir over-325kb
    find . -type f -maxdepth 1 -size -325k -name "*.wav" -exec mv -t under-325kb {} +
    find . -type f -maxdepth 1 -size +325k -name "*.wav" -exec mv -t over-325kb {} +
    
    # copy all the .txt files from the parent folder
    find . -type f -name "*.wav" -exec sh -c 'base=`basename {} .wav`; mv ../${base}.txt .' \;
    
    When using split limit 325kb, the sequence length with default MFCC
    parameters is about 1035 for audio files that are in the under partition.
    '''
