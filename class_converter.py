from datetime import *
import pandas as pd
from dateutil.parser import parse
from StringIO import StringIO
import os
import os.path
import io
from io import *
from itertools import chain
import io
from io import *
import pandas as pd
from numpy import *
from pandas import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pytz
import sys


def get_day_number(day):
	day = int(day)
	list_of_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	list_of_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
	sum = 0
	k = 0
	while sum < day and sum < 730:
		sum+=list_of_days[k]
		k+=1
	k = k -1
	sum -=list_of_days[k]
	date = day - sum
	month = list_of_months[k]
	if k <= 12:
		year = 2009
	else:
		year = 2010
	if (k <12):
		return (str(year)+"-"+str(k+1)+"-"+str(date))
	else:
		return (str(year)+"-"+str(k-11)+"-"+str(date))
	#return datetime(year, month, date)
def get_time(digits):
	if (int(digits) > 48):
		digits = str(int(digits) % 48) 
	list_of_keys = [str(i) for i in range(1, 49)]
	#print list_of_keys
	index = pd.date_range(start = '2009-March-7', periods = 48, freq = '30min')
	list_of_values = [str(i)[10:] for i in index]
	dic = {}
	for i in range(len(list_of_keys)):
		dic[list_of_keys[i]] = list_of_values[i]
	return dic[digits]
def generate_timestamp_as_datetime(class_timestamp):
	string = generate_timestamp_as_string(class_timestamp)
	print string
	return parse(string)
def generate_timestamp_as_string(class_timestamp):
	first_part = class_timestamp/100
	second_part = class_timestamp % 100
	timestamp = get_day_number(first_part)+get_time(str(second_part))
	return timestamp
def get_dataset(path_to_txt):
	f1 = open(path_to_txt, "r")
	data = u"ID Code Power\n" + str(f1.read())
	TESTDATA = StringIO(data)
	df2 = DataFrame.from_csv(TESTDATA, sep=" ", parse_dates=False)
	return df2[:336]
def get_processed_dataset(path_to_txt):
	dataset = get_dataset(path_to_txt)
	id = dataset.index
	codes = dataset["Code"].values
	power = dataset["Power"].values
	dataset["Code"] = np.asarray([generate_timestamp_as_datetime(i) for i in codes])
	return dataset
def construct_new_dataframe(path_to_txt):
	dataset = get_processed_dataset(path_to_txt)
	index = dataset.index
	code = np.asarray(dataset["Code"])
	power = np.asarray(dataset["Power"])
	index, code = code, index
	df = DataFrame(code, index)
	df["Power"] = power
	return df
def get_features(path_to_txt):
	dataset = construct_new_dataframe(path_to_txt)
	#Consumption figures. Come on man. You can fucking do this.
	vector_1 = dataset["Power"].values
	temp = pd.DatetimeIndex(dataset.index)
	dataset["Weekday"] = dataset["Power"][temp.weekday == 1]
	dataset["Weekend"] = dataset["Power"][temp.weekday == 3]
	dataset["Maximum"] = df["Power"].resample('1d', how = 'max')
	dataset["Minimum"] = df["Power"].resample('1d', how = 'min')
	df["Between 6 pm and 10 pm"] = df["Power"].between_time("18:00", "22:00")
	df["Between 6 am and 10 am"] = df["Power"].between_time("6:00", "10:00")
	df["Between 1 am and 5 am"] = df["Power"].between_time("1:00", "5:00")
	df["Between 10 am and 2m"] = df["Power"].between_time("10:00", "14:00")
	print dataset.head(10)
get_features("/Users/Rishi/Downloads/file1.txt")
#print construct_new_dataframe("/Users/Rishi/Downloads/file1.txt").head()
#FUCK YOU LIFE!!!