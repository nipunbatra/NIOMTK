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
from sklearn import svm, datasets, metrics
import random
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from common_functions import  *
import itertools
from sklearn import tree
random.seed(42)


classifiers_dict = {"SVM": svm.SVC(), "KNN": KNeighborsClassifier(n_neighbors=3),
                    "DT": tree.DecisionTreeClassifier(),"RF":RandomForestClassifier()}

metric_dict = {"Precision":precision_score, "Recall":recall_score,
               "MCC":matthews_corrcoef, "Accuracy":accuracy_score}



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
# def get_features_new(path_to_txt):
# 	dataset = get_processed_dataset(path_to_txt)
# 	feature_array = []
# 	for i in list(dataset.index):
# 		temp_dataset = dataset[i]
# 		feature_array.append(get_features(temp_dataset))
# 	print feature_array


def get_features(path_to_txt):
	dataset = construct_new_dataframe(path_to_txt)['2009-07-14':'2009-07-21']
	dataset = dataset.sort()
	#print dataset.head()
	vector_1 = dataset["Power"].values # c_day
	temp = pd.DatetimeIndex(dataset.index)
	print "Temp.weekday", temp.weekday
	dataset["Weekday"] = dataset["Power"][temp.weekday < 5] #c_weekday
	dataset["Weekend"] = dataset["Power"][temp.weekday >= 5] #c_weekend
	print dataset.head(15)
	dataset["Maximum"] = dataset["Power"].resample('1d', how = 'max') #c_max
	dataset["Minimum"] = dataset["Power"].resample('1d', how = 'min') #c_min
	dataset["Between 6 pm and 10 pm"] = dataset["Power"].between_time("18:00", "22:00") # c_evening
	dataset["Between 6 am and 10 am"] = dataset["Power"].between_time("6:00", "10:00") # c_morning
	dataset["Between 1 am and 5 am"] = dataset["Power"].between_time("1:00", "5:00") # c_night
	dataset["Between 10 am and 2m"] = dataset["Power"].between_time("10:00", "14:00") # c_noon
	#Temporal properties
	dataset["Entry greater than 1 KW"] = dataset["Power"][dataset["Power"] > 1]
	dataset["Entry greater than 2 KW"] = dataset["Power"][dataset["Power"] > 2]
	dataset["Entry equal to maximum"] = dataset["Power"][dataset["Power"] == dataset["Power"].max()]
	dataset["Entry greater than mean"] = dataset["Power"][dataset["Power"] > dataset["Power"].mean()]
	dataset = dataset.fillna(value = 0)
	#ratios
	

	return dataset
	print "Length of dataset = ", len(dataset)

def compute(path_to_txt):
	employed = []
	df2 = get_features(path_to_txt)
	df2 = df2.fillna(value = 0)
	fp = get_employment_status("/Users/Rishi/Downloads/abc.csv")
	fp = fp[df2["Code"].values]
	fp = fp.fillna(value = 0)
	lis = fp.values
	df = df2.drop("Code", axis = 1)
	df = df.drop("Index", axis = 1)
	df = (df + 0.0)
	df2_train = df.head(len(df2)/2)

	ground_truth_train = lis[:len(df)/2]
	df2_test = df.tail(len(df)/2)
	ground_truth_test = lis[len(df)/2:]
	dataframe_train_array = np.asarray([i for i in df2_train.values])
	ground_truth_train_array = np.asarray(ground_truth_train)
	dataframe_test_array = np.asarray([i for i in df2_test.values])
	ground_truth_test_array = np.asarray(ground_truth_test)

	print ground_truth_test_array
	out = {}
	for clf_name, clf in classifiers_dict.iteritems():
		out[clf_name] = {}
		clf.fit((dataframe_train_array[:100000]),(ground_truth_train_array[:100000]))
		prediction = clf.predict(dataframe_test_array[:100000])
		print "Precision for ", clf_name, " is ", precision_score(ground_truth_test, prediction)
		print "Recall for ", clf_name, " is ", recall_score(ground_truth_test, prediction)
		#print "MCC for ", clf_name, " is ", matthews_corrcoef(ground_truth_test, prediction)
		print "Accuracy for ", clf_name, " is ", accuracy_score(ground_truth_test, prediction)
		#for metric_name, metric_func in metric_dict.iteritems():
		#	out[clf_name][metric_name] = metric_func(ground_truth_test, prediction)
	output_dataframe = pd.DataFrame(out)
	print output_dataframe


	#classifier = svm.SVC().fit((dataframe_train_array[:100000]),(ground_truth_train_array[:100000]))
	#prediction = classifier.predict(dataframe_test_array[:100000])
	#print (metrics.classification_report(prediction, ground_truth_test[:100000]))
	# for i in list(df2["Code"]):
	# 	if (fp[i] < 4):
	# 		employed.append(1)
	# 	else:
	# 		employed.append(0)
	print "Lengths: ", len(df2), len(fp)
# def get_features(dataset):
# 	#dataset = construct_new_dataframe(path_to_txt)['2009-07-14':'2009-07-21']
# 	#Consumption figures
# 	c_day = dataset["Power"].mean()
#  	temp = pd.DatetimeIndex(dataset.index)
#  	print "Temp.weekday", temp.weekday
#  	dataset["Weekday"] = dataset["Power"][temp.weekday < 5] #c_weekday
#  	dataset["Weekend"] = dataset["Power"][temp.weekday >= 5]
#  	dataset = dataset.fillna(value = 0)
#  	c_weekday = dataset["Weekday"].mean()
#  	c_weekend = dataset["Weekend"].mean()
#  	dataset["Maximum"] = dataset["Power"].resample('1d', how = 'max') #c_max
# 	dataset["Minimum"] = dataset["Power"].resample('1d', how = 'min') #c_min
# 	dataset["Between 6 pm and 10 pm"] = dataset["Power"].between_time("18:00", "22:00") # c_evening
# 	dataset["Between 6 am and 10 am"] = dataset["Power"].between_time("6:00", "10:00") # c_morning
# 	dataset["Between 1 am and 5 am"] = dataset["Power"].between_time("1:00", "5:00") # c_night
# 	dataset["Between 10 am and 2 pm"] = dataset["Power"].between_time("10:00", "14:00") # c_noon
# 	c_max = dataset["Power"].max()
# 	c_min = dataset["Power"].min()
# 	c_evening = dataset["Between 6 pm and 10 pm"].mean()
# 	c_morning = dataset["Between 6 am and 10 am"].mean()
# 	c_night = dataset["Between 1 am and 5 am"].mean()
# 	c_noon = dataset["Between 1 am and 5 am"].mean()
# 	return [c_day, c_weekday, c_max, c_min, c_evening, c_morning, c_night, c_noon]


#print get_features_new("/Users/Rishi/Downloads/file1.txt")

def get_csv_ground_truth(path_to_csv):
	df = pd.read_csv(path_to_csv)
	return df

def get_number_of_bedrooms(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	return df["Question 460: How many bedrooms are there in your home"]

def get_type_of_cooking(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	return df["Question 4704: Which of the following best describes how you cook in your home"]

def get_employment_status(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 310: What is the employment status of the chief income earner in your household, is he/she"][df["Question 310: What is the employment status of the chief income earner in your household, is he/she"] < 4]
	dataset = dataset.fillna(value = 0)
	return dataset

def get_retirement_status(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 310: What is the employment status of the chief income earner in your household, is he/she"][df["Question 310: What is the employment status of the chief income earner in your household, is he/she"] == 5]
	dataset = dataset.fillna(value = 0)
	return dataset

def get_family(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 410: What best describes the people you live with? READ OUT"][df["Question 410: What best describes the people you live with? READ OUT"]> 1]
	dataset = dataset.fillna(value = 0)
	return dataset

compute("/Users/Rishi/Downloads/file1.txt")
#print construct_new_dataframe("/Users/Rishi/Downloads/file1.txt").head(40)
