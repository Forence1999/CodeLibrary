# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022
import cv2


def get_video_duration_opencv(filepath):
	cap = cv2.VideoCapture(filepath)
	rate = cap.get(5)
	frame_num = cap.get(7)
	duration = frame_num / rate
	return duration


if __name__ == '__main__':
	print('Hello World!')

	print('Brand-new World!')
