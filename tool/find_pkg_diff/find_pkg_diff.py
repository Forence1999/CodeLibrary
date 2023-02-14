import warnings


def collect_pkg(path):
	'''
	Collect all the packages in the file.
	:param path:
	:return:
	'''
	with open(path, 'r') as f:
		lines = f.readlines()
		pkg = set()
		for line in lines:
			line = line.strip('\n').strip(' ').split(' ')
			line = [i for i in line if i != '']
			if len(line) == 3:
				pkg.add(line[0])
				# pkg.add(line[0]+' = '+line[1])
			elif len(line) == 4:
				pkg.add(line[0])
				# pkg.add(line[0]+' = '+line[1])
				# pkg.add(line[0] + '=' + line[1] + ' ' + line[3])
			else:
				warnings.warn('The line is not correct: {}'.format(line))
	return pkg


if __name__ == '__main__':

	pkg_15 = collect_pkg('./g15.txt')
	pkg_14 = collect_pkg('./g14.txt')

	print('g14: ', pkg_14 - pkg_15)
	print()
	print('g15: ', pkg_15 - pkg_14)

	# res = pkg_15 - pkg_14
	# res = ["pip install " + i.strip(' pypi') for i in res]
	# print(" && ".join(res))
	#
	# pass

# && pip install apex==0.1
# && pip install aml==6.0
# && pip install arsing==3.0.9
# && pip install ackaging==21.3
# pip install colorama==0.4.5 && pip install tabulate==0.8.10 && pip install decorator==5.1.1 && pip install appdirs==1.4.4 && pip install tqdm==4.64.1 && pip install lxml==4.9.1 && pip install cython==0.29.32
