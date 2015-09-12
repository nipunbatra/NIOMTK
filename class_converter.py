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
from pandas.tseries.offsets import *
import random
random.seed(42)
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
	print "Parsed string: ", parse(string)
	print "Parsed string with type string:", str(parse(string))
	print "Date offset: ", Second(random.uniform(0, 1801))
	#print "Parsed string + DateOffset: ", str(parse(string)  + Second(0, 1801) + Milli(random.uniform(0, 1000)))
	temp = parse(str(parse(string)  + Second(random.random() * 1800) + Milli(random.random() * 1800)))
	print "Datetime: ", temp
	return temp
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
	print "PRINT THE FUCKING HEAD", df2.head()
	df2.sort()
	return df2[:1000]
def get_processed_dataset(path_to_txt):
	dataset = get_dataset(path_to_txt)
	id = dataset.index
	codes = dataset["Code"].values
	power = dataset["Power"].values
	dataset["Code"] = np.asarray([generate_timestamp_as_datetime(i) for i in codes])
	print dataset.head()
	return dataset
def construct_new_dataframe(path_to_txt):
	dataset = get_processed_dataset(path_to_txt)
	#print dataset.head()
	#dataset.set_index(['Code'])
	index = dataset.index
	code = np.asarray(dataset["Code"])
	power = np.asarray(dataset["Power"])
	index, code = code, index
	dic = {"Index":index, "Code": code, "Power": power}
	df = pd.DataFrame(dic)
	df = df.set_index([pd.DatetimeIndex(df["Index"])])
	# df = DataFrame(code, index)
	# df["Power"] = power
	# df.resample('D', how = 'max')
	return df.sort_index()
def get_features(path_to_txt):
	dataset = construct_new_dataframe(path_to_txt)['2009-07-14':'2009-07-21']
	dataset = dataset.sort()
	#print dataset.head()
	#Consumption figures. Come on man. You can fucking do this.
	vector_1 = dataset["Power"].values
	temp = pd.DatetimeIndex(dataset.index)
	print "Temp.weekday", temp.weekday
	dataset["Weekday"] = dataset["Power"][temp.weekday < 5]
	dataset["Weekend"] = dataset["Power"][temp.weekday >= 5]
	print dataset.head(15)
	dataset["Maximum"] = dataset["Power"].resample('1d', how = 'max')
	dataset["Minimum"] = dataset["Power"].resample('1d', how = 'min')
	dataset["Between 6 pm and 10 pm"] = dataset["Power"].between_time("18:00", "22:00")
	dataset["Between 6 am and 10 am"] = dataset["Power"].between_time("6:00", "10:00")
	dataset["Between 1 am and 5 am"] = dataset["Power"].between_time("1:00", "5:00")
	dataset["Between 10 am and 2m"] = dataset["Power"].between_time("10:00", "14:00")
	#Temporal properties
	dataset["Entry greater than 1 KW"] = dataset["Power"][dataset["Power"] > 1]
	dataset["Entry greater than 2 KW"] = dataset["Power"][dataset["Power"] > 2]
	dataset["Entry equal to maximum"] = dataset["Power"][dataset["Power"] == dataset["Power"].max()]
	dataset["Entry greater than mean"] = dataset["Power"][dataset["Power"] > dataset["Power"].mean()]
	return dataset
	#print "Length of dataset = ", len(dataset)

print get_features("/Users/Rishi/Downloads/file1.txt")
#print construct_new_dataframe("/Users/Rishi/Downloads/file1.txt").head(40)
#FUCK YOU LIFE!!!