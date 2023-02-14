# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022
import json
import os


# -------------------------- json -------------------------- #

def json_writer(data, path):  # TODO: has not been tested
	data = json.dumps(data)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as writer:
		writer.write(data)


if __name__ == '__main__':
	print('Hello World!')

	print('Brand-new World!')
