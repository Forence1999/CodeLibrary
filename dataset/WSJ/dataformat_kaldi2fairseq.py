# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 28/3/2023

import re
from pathlib import Path
from analysis.CodeLibrary.audio import get_audio_frame_num
from analysis.CodeLibrary.path import convert_extension, convert_rootpath


def scp2tsv(scp_path, tsv_path):
    '''
    convert kaldi *.scp to fairseq *.tsv
    :param scp_path: the path of *.scp
            scp_file:
                id path (e.g. 4k0a0106 /data2/swang/asr/wsj/raw/wsj1/13-16.1/wsj1/si_dt_05/4k0/4k0a0106.wv1 )

    :param tsv_path: the path of *.tsv
            tsv_file:
                root_path (e.g. /data/swang/asr/LibriSpeech/dev-other)
                path \t frame_num (e.g. 4572/112383/4572-112383-0009.flac	100480)
    :return:
    '''

    global raw_dsroot, flac_dsroot

    scp_path = Path(scp_path)
    tsv_path = Path(tsv_path)

    # get path_ls and frame_num_ls
    lines = scp_path.read_text().splitlines()
    path_ls = [i.split()[1] for i in lines if i]
    path_ls = convert_extension(convert_rootpath(path_ls, raw_dsroot, flac_dsroot), '.flac')
    frame_num_ls = [get_audio_frame_num(i) for i in path_ls]

    # write results to tsv file
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open('w') as f:
        print(str(flac_dsroot), file=f)
        for path, frame_num in zip(path_ls, frame_num_ls):
            print(f'{str(path.relative_to(flac_dsroot))}\t{frame_num}', file=f)


def normalize_sentence(sentence):
    '''
    normalize sentence: remove <NOISE>, remove non-letter and non-apostrophe, remove extra spaces, upper case, strip
    :param sentence: the sentence to be normalized
    :return: the normalized text
    '''
    sentence = sentence.replace("<NOISE>", " ")
    assert "<" not in sentence and ">" not in sentence, f" \"<\" or \">\" in sentence: {sentence}"
    sentence = re.sub(r"[^a-zA-Z ']", " ", sentence)  # only keep letters, apostrophes and spaces
    sentence = re.sub(r" +", ' ', sentence).upper().strip()

    return sentence


def wrd2ltr(sentence):
    if sentence == "":
        return ""
    else:
        return " ".join(list(sentence.replace(" ", "|"))) + " |"


def txt2wrd2ltr(txt_path, wrd_path, ltr_path):
    '''
    convert kaldi *.txt to fairseq *.wrd and *.ltr
    :param txt_path: the path of *.txt
            txt_file:
                id sentence (e.g. 4k0a0101 he ceased <NOISE> )
    :param wrd_path: the path of *.wrd
            wrd_file:
                normailized_sentence (e.g. HE CEASED )
    :param ltr_path: the path of *.ltr
            ltr_file:
                discretized_sentence (e.g. H E | C E A S E D | )
    :return:
    '''

    txt_path = Path(txt_path)
    wrd_path = Path(wrd_path)
    ltr_path = Path(ltr_path)

    # get sentence_ls
    lines = txt_path.read_text().splitlines()
    sentence_ls = [i.split(maxsplit=1)[1] for i in lines if i]

    # generate the desired wrd and ltr files
    wrd_ls = [normalize_sentence(i) for i in sentence_ls]
    ltr_ls = [wrd2ltr(i) for i in wrd_ls]

    # write results to wrd and ltr files
    with wrd_path.open('w') as wrd_file, \
            ltr_path.open('w') as ltr_file:
        for wrd, ltr in zip(wrd_ls, ltr_ls):
            print(wrd, file=wrd_file)
            print(ltr, file=ltr_file)


raw_dsroot = Path('/data2/swang/asr/wsj/raw')
flac_dsroot = Path('/data2/swang/asr/wsj/flac')

if __name__ == '__main__':
    crt_dir = Path('./').absolute()
    data_dir = Path('./data').absolute()
    processed_dir = Path('./processed').absolute()

    # process *_sph.scp
    for scp_path in data_dir.glob('*_sph.scp'):
        # generate tsv file path
        tsv_dir = processed_dir / Path(scp_path).relative_to(data_dir).parent
        tsv_base = Path(scp_path).stem.rstrip('_sph') + '.tsv'
        tsv_path = tsv_dir / tsv_base

        scp2tsv(scp_path, tsv_path)

    # process *.txt
    for txt_path in data_dir.glob('*.txt'):
        # generate wrd and ltr file path
        wrd_path = processed_dir / Path(txt_path).relative_to(data_dir).with_suffix('.wrd')
        ltr_path = processed_dir / Path(txt_path).relative_to(data_dir).with_suffix('.ltr')

        txt2wrd2ltr(txt_path, wrd_path, ltr_path)
