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


def multiprocess_Pool_async(func, argument_collection, num_workers=8):
	'''
	using multiprocessing.Pool to perform func on the collection
	:param func: function to perform
	:param argument_collection: collection of func arguments. list like
			[
				{
					"args"          : (),
					"kwds"          : {},
					"callback"      : None,
					"error_callback": None
				},
				...
			]
	:param num_workers: number of workers
	:return:
	'''
	
	# with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
	with multiprocessing.Pool(processes=num_workers) as pool:
		results = []
		# Apply the process_task function to each task asynchronously
		for i, argument in enumerate(argument_collection):
			result = pool.apply_async(func=func, **argument)
			results.append(result)
		
		final_results = [result.get() for result in results]  # Wait for all tasks to complete and collect the results
	
	return final_results
