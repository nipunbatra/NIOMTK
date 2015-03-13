import numpy as np
import pandas as pd
import matplotlib
import os
import os.path
from os import *
from os.path import *

def get_files(dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK'):
	assert isdir(dir_path)
	files=[join(dir_path, file) for file in listdir(dir_path) if isfile(join(dir_path, file)) and '.h5' and file]
	files.sort()
	return files

def get_daily_average_for_all(dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK'):
	files=get_files(dir_path)
	for i in files:
		current_file=join(dir_path, i)
		get_average_for_current(current_file)


def get_average_for_current(current_file):
	store=HDFStore(current_file)
	for key in store.keys():
		value=store[key] # getting the key from the store
		# at this point, value stores the unsampled dataset
		data_daily_average=value.resample('D', how='mean')
		threshold=get_threshold(data_daily_average, key)

def get_std_for_current(current_file):
	store=HDFStore(current_file)
        for key in store.keys():
                value=store[key] # getting the key from the store
                # at this point, value stores the unsampled dataset
                data_daily_average=value.resample('D', how=np.std)


