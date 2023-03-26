# -*- coding:utf-8 _*-
# @License: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 26/3/2023

import os, sys
import time
import random
import warnings
import numpy as np
from pathlib import Path
import multiprocessing


def multiprocess_Pool(func, collection, num_workers=8):
	'''
	using multiprocessing.Pool to perform func on the collection
	:param func: function to perform
	:param collection: collection of data (elements)
	:param num_workers: number of workers
	:return:
	'''
	
	with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
		results = []
		# Apply the process_task function to each task asynchronously
		for i, element in enumerate(collection):
			result = pool.apply_async(func=func, args=(element,))
			results.append(result)
		
		final_results = [result.get() for result in results]  # Wait for all tasks to complete and collect the results
	
	return final_results

# threads = []
#
# for arrayval in array:
#
#     threads.append(Thread(target=dosomething, args=(arrayval,)))
#
#     threads[-1].start()
#
# for thread in threads:
#
#     thread.join()
#
# from threading import Thread
#
#
# def say(name):
#
#     pass
#
#
# def listen(name):
#
#     pass
#
#
# if __name__ == '__main__':
#
#     t1 = Thread(target=say, args=('tony',))
#
#     t1.start()
#
#     t2 = Thread(target=listen, args=('simon',))
#
#     t2.start()
#
# print("程序结束=====================")


# from multiprocessing import Pool
#
# pool = Pool(processes=num_workers - 1)
# worker_results = [
# 	pool.apply_async(
# 		func,
# 		args=(
# 			...
# 		),
# 		kwds={
# 			...
# 		}
# 		if ...
# 		else {},
# 	)
# 	for worker_id, (start_offset, end_offset) in enumerate(
# 		more_chunks, start=1
# 	)
# ]
#
# pool.close()
# pool.join()

# from multiprocessing import Process
#
#
# processes = []
#
# for fold in range(folds):
#
#     processes.append(Process(target=single_process, args=(fold,)))
#
#     processes[-1].start()
#
# for process in processes:
#
#     process.join()
