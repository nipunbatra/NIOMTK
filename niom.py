import numpy as np
import pandas as pd
import matplotlib
import os
import os.path
from os import *
from os.path import *
from pandas import *

'''def get_files(dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets'):
	assert isdir(dir_path)
	files=[join(dir_path, file) for file in listdir(dir_path) if isfile(join(dir_path, file)) and 'combed.h5' in file]
	files.sort()
	return files

def get_daily_average_for_all(dir_path='/Users/rishi/Documents/Master_folder/IIITD/6th_semester/BTP/NIOMTK_datasets'):
	files=get_files(dir_path)
	for i in files:
		current_file=join(dir_path, i)
		get_average_for_current(current_file)


def get_average_for_current(current_file):
	store=HDFStore(current_file)
	for key in store.keys():
		value=store[key] # getting the key from the store
		# at this point, value stores the unsampled dataset
		data_daily_average=value.resample('15min', how='mean')
		threshold=get_threshold(data_daily_average)
		print ("Printing results only according to the mean: ")
		print (data_daily_average > threshold)
		print ("Printing results only according to threshold calculated by using averages: ")
		average_threshold=get_average_threshold(data_daily_average)
		print (data_daily_average > average_threshold)

def get_std_for_current(current_file):
	store=HDFStore(current_file)
        for key in store.keys():
                value=store[key] # getting the key from the store
                # at this point, value stores the unsampled dataset
                data_daily_std=value.resample('15min', how=np.std)
		threshold=get_threshold (data_daily_average)
		print ("Printing results only according to standard deviation: ")
		print (data_daily_std > threshold)

def get_range_for_current(current_file):
	store=HDFStore(current_file)
	for key in store.keys():
		value=store[key]
		data_daily_range=value.resample('D')
		for i in data_daily_range.index:
		
def get_threshold(dataframe):
	arr=dataframe.values
	current_max=arr[0][1]
	threshold=arr[0]
	for i in range(1, len(arr)):
		if (arr[i][1])>current_max:
			current_max=arr[i][1]
			threshold=arr[i]
	return threshold

def find_max(dataframe):
        arr=dataframe.values
        current_max=arr[0][1]
        threshold=arr[0]
        for i in range(1, len(arr)):
                if (arr[i][1])>current_max:
                        current_max=arr[i][1]
                        threshold=arr[i]
        return threshold

def find_min(dataframe):
        arr=dataframe.values
        current_min=arr[0][1]
        threshold=arr[0]
        for i in range(1, len(arr)):
                if (arr[i][1]) < current_min:
                        current_min=arr[i][1]
                        threshold=arr[i]
        return threshold


def get_average_threshold(dataframe):
	threshold=sum(dataframe.values)/len(dataframe.values)
	return threshold

get_files()
get_daily_average_for_all()'''
