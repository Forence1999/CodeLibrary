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


def normalize_sentence(sentence):
	'''
	normalize the sentence: delete the content in brackets, only keep letters and apostrophes, remove extra spaces, upper case, strip
	:param sentence: the sentence to be normalized
	:return: the normalized text
	'''
	sentence = re.sub(r'\[.*?\]', ' ', sentence)
	sentence = re.sub(r"[^a-zA-Z ']", " ", sentence)  # only keep letters, apostrophes and spaces
	sentence = re.sub(r" +", ' ', sentence).upper().strip()
	
	return sentence


def wrd2ltr(sentence):
	if sentence == "":
		return ""
	else:
		return " ".join(list(sentence.replace(" ", "|"))) + " |"


def txt2tsv_wrd_ltr(txt_path, tsv_path, wrd_path, ltr_path):
	'''
	convert kaldi text to fairseq all.wrd and all.ltr
	:param txt_path: the path of *.txt
			txt_file:
				id sentence (e.g. 4k0a0101 he ceased <NOISE> )
	:param tsv_path: the path of *.tsv
			tsv_file:
				root_path (e.g. /data/swang/asr/LibriSpeech/dev-other)
				path \t frame_num (e.g. 4572/112383/4572-112383-0009.flac	100480)
	:param wrd_path: the path of *.wrd
			wrd_file:
				normailized_sentence (e.g. HE CEASED )
	:param ltr_path: the path of *.ltr
			ltr_file:
				discretized_sentence (e.g. H E | C E A S E D | )
	:return:
	'''
	global flac_dsroot
	
	txt_path = Path(txt_path)
	tsv_path = Path(tsv_path)
	wrd_path = Path(wrd_path)
	ltr_path = Path(ltr_path)
	
	# generate flac dict: basename -> path
	flac_ls = flac_dsroot.rglob('*.flac')
	flac_dict = {i.stem: i for i in flac_ls}
	
	# get segments
	segments = txt_path.read_text().splitlines()
	segments = [i.split(maxsplit=1) for i in segments if i]
	data = [(flac_dict[i[0]], i[1]) for i in segments]  # list of (path, sentence)
	
	# write results to tsv, wrd and ltr files
	tsv_path.parent.mkdir(parents=True, exist_ok=True)
	wrd_path.parent.mkdir(parents=True, exist_ok=True)
	ltr_path.parent.mkdir(parents=True, exist_ok=True)
	with tsv_path.open('w') as tsv_file, wrd_path.open('w') as wrd_file, ltr_path.open('w') as ltr_file:
		print(str(flac_dsroot.absolute()), file=tsv_file)
		
		for path, sentence in data:
			sentence = normalize_sentence(sentence)
			if sentence == "":
				continue
			
			# write wrd and ltr files
			print(sentence, file=wrd_file)
			print(wrd2ltr(sentence), file=ltr_file)
			
			# write tsv file
			frame_num = get_audio_frame_num(path)
			print(f'{str(path.relative_to(flac_dsroot))}\t{frame_num}', file=tsv_file)


def normalize_file(src_path, dst_path):
	'''
	Delete the content in brackets for each sentence
	:param src_path:
	:param dst_path:
	:return:
	'''
	
	src_path = Path(src_path)
	dst_path = Path(dst_path)
	
	line_ls = src_path.read_text().splitlines()
	line_ls = [line.split(maxsplit=1) for line in line_ls]
	
	# delete the content in brackets
	line_ls = [[line[0], normalize_sentence(line[1])] for line in line_ls]
	line_ls = [line[0] + " " + line[1] for line in line_ls if line[1] != ""]
	
	dst_path.parent.mkdir(parents=True, exist_ok=True)
	dst_path.unlink(missing_ok=True)
	dst_path.write_text("\n".join(line_ls) + "\n")


flac_dsroot = Path('/data2/swang/asr/swb/flac')

if __name__ == '__main__':
	crt_dir = Path('./').absolute()
	data_dir = Path('./train').absolute()
	processed_dir = Path('./processed').absolute()
	
	src_path = Path("./train/text").absolute()
	tgt_path = Path("./processed/norm_text").absolute()
	
	# # normalize the text file (text labels)
	# normalize_file(src_path, tgt_path)
	
	# generate tsv wrd and ltr files
	txt2tsv_wrd_ltr(src_path, processed_dir / 'all.tsv', processed_dir / 'all.wrd', processed_dir / 'all.ltr')
