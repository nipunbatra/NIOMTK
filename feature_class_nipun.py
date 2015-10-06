from datetime import *
import pandas as pd
from dateutil.parser import parse
from StringIO import StringIO
import os
import os.path
import io
from itertools import chain
from io import *
import pandas as pd
from numpy import *
from pandas import *
from sklearn.neighbors import KNeighborsClassifier
import pytz
import sys
from pandas.tseries.offsets import *
from itertools import chain
from numpy import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pytz
import sys
from sklearn import svm, datasets, metrics
import random
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from common_functions import *
import itertools
from sklearn import tree
from numpy import inf
from os import *
from os.path import *

random.seed(42)

classifiers_dict = {"SVM": svm.SVC(), "KNN": KNeighborsClassifier(n_neighbors=3),
                    "DT": tree.DecisionTreeClassifier(), "RF": RandomForestClassifier()}

metric_dict = {"Precision": precision_score, "Recall": recall_score, "Accuracy": accuracy_score}


def get_day_number(day):
    day = int(day)
    list_of_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    list_of_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                      "November", "December", "January", "February", "March", "April", "May", "June", "July", "August",
                      "September", "October", "November", "December"]
    sum = 0
    k = 0
    while sum < day and sum < 730:
        sum += list_of_days[k]
        k += 1
    k = k - 1
    sum -= list_of_days[k]
    date = day - sum
    month = list_of_months[k]
    if k <= 12:
        year = 2009
    else:
        year = 2010
    if (k < 12):
        return (str(year) + "-" + str(k + 1) + "-" + str(date))
    else:
        return (str(year) + "-" + str(k - 11) + "-" + str(date))
        # return datetime(year, month, date)


def get_time(digits):
    if (int(digits) > 48):
        digits = str(int(digits) % 48)
    # list_of_keys = [('0' + str(i)) for i in range(1, 10)] + [str(i) for i in range(10, 49)]
    list_of_keys = [str(i) for i in range(1, 49)]
    index = pd.date_range(start='2009-March-7', periods=48, freq='30min')
    list_of_values = [str(i)[10:] for i in index]
    dic = {}
    for i in range(len(list_of_keys)):
        dic[list_of_keys[i]] = list_of_values[i]
    return dic[digits]


def generate_timestamp_as_string(class_timestamp):
<<<<<<< HEAD
    first_part = int(class_timestamp) / 100
    second_part = int(class_timestamp) % 100
    timestamp = get_day_number(first_part) + get_time(str(second_part))
    return timestamp


def get_dataset_concat(path_to_txt):
    # edf = pd.read_csv(path_to_txt, sep=" ", nrows=200000, names=["home", "hour", "Power"])
    input_frames = []
    files = [f for f in listdir(path_to_txt) if isfile(join(path_to_txt,
                                                            f)) and '.txt' in f]
    print "Reading files..."
    for i in files:
        print "File being read: ", i
        edf = pd.read_csv(join(path_to_txt, i), sep=" ", nrows=200000, names=["home", "hour", "Power"])
        input_frames.append(edf)
    print "Done!"
    edf = concat(input_frames)
    return edf


def create_feature_matrix(edf, survey_csv):
    print edf["hour"]
    list_of_ids = edf["home"].unique()[:-1]
    list_of_frames = []
    out_dict = {}
    num_homes = len(list_of_ids)
    print "TOTAL", num_homes
    for home_id in list_of_ids:
        print home_id

        out_dict[home_id] = {}
        start_date = (edf["hour"][edf.home == home_id].iloc[0])
        a = edf[(edf.home == home_id) & (edf.hour > 19500) & (edf.hour < 20248)].Power
        a.index = pd.DatetimeIndex(start=generate_timestamp_as_string(start_date), freq='30T', periods=len(a))
        df = pd.DataFrame(a, columns=["Power"])
        df["Code"] = home_id
        out_dict[home_id] = get_features(df)
    list_of_features = ["c_day", "c_weekday", "c_weekend", "c_max", "c_min", "c_evening", "c_morning", "c_night",
                        "c_noon", "r_mean", "r_min", "r_night", "r_morning", "r_evening", "s_variance", "s_diff",
                        "s_num_peaks", "s_x_corr"]
    dataset = pd.DataFrame(out_dict)  # , index = list_of_ids, columns = list_of_features)
    dataset = dataset.T
    dataset.index = list_of_ids
    dataset.columns = list_of_features
    # print "OUT DICT: ", out_dict
    fp = get_retirement_status(survey_csv)
    fp = fp[list_of_ids]
    dataset["Retirement_answers"] = fp
    dataset = dataset.fillna(value=0)
    return dataset

def evaluate(dataset):
    # result = concat(list_of_frames)
    # print result.head()
    # print len(result)
    compute(dataset)

=======
	print class_timestamp
	first_part = int(class_timestamp)/100
	second_part = int(class_timestamp) % 100
	timestamp = get_day_number(first_part)+get_time(str(second_part))
	return timestamp
def get_dataset(path_to_txt):
	#edf = pd.read_csv(path_to_txt, sep=" ", nrows=200000, names=["home", "hour", "Power"])
	input_frames = []
	files = [f for f in listdir(path_to_txt) if isfile(join(path_to_txt, f)) and '.txt' in f]# and 'File2' not in f and 'File3' not in f and 'File6' not in f and 'File5' not in f and 'File4' not in f]
	print "Reading files..."
	for i in files:
		print "File being read: ", i
		edf = pd.read_csv(join(path_to_txt, i), sep=" ",nrows = 2000000, names=["home", "hour", "Power"])
		input_frames.append(edf)
	print "Done!"
	edf = concat(input_frames)
	print edf["hour"]
	list_of_ids = edf["home"].unique()
	if (len(list_of_ids) % 2 == 1):
		list_of_ids = list_of_ids[:-1]
	list_of_frames = []
	out_dict = {}
	print len(list_of_ids)
	for home_id in list_of_ids:
		out_dict[home_id] = {}
		start_date = (edf["hour"][edf.home == home_id].iloc[0])
		a=edf[(edf.home==home_id)&(edf.hour>19500)&(edf.hour<20248)].Power
		a.index = pd.DatetimeIndex(start=generate_timestamp_as_string(start_date), freq='30T', periods=len(a))
		df = pd.DataFrame(a, columns=["Power"])
		df["Code"] = home_id
		out_dict[home_id] = get_features(df)
		print out_dict
	list_of_features = ["c_day", "c_weekday", "c_weekend", "c_max", "c_min", "c_evening", "c_morning", "c_night", "c_noon", "r_mean", "r_min", "r_night", "r_morning", "r_evening", "s_variance", "s_diff", "s_num_peaks", "s_x_corr"]
	dataset = pd.DataFrame(out_dict)#, index = list_of_ids, columns = list_of_features)
	dataset = dataset.T
	dataset.index =list_of_ids
	dataset.columns = list_of_features
	#print "OUT DICT: ", out_dict
	fp = get_retirement_status("/Users/Rishi/Downloads/abc.csv")
	print fp.head()
	#fp = fp[list_of_ids]
	print fp
	dataset["Retirement_answers"] = fp
	dataset = dataset.fillna(value = 0)
	print dataset
	dataset.to_hdf('cer.h5', 'data')
	# result = concat(list_of_frames)
	# print result.head()
	# print len(result)
	compute(dataset)
>>>>>>> 468256f9912089df156d7aea850bd85b52ea699a

def get_features(dataset):
    c_day = dataset["Power"].mean()
    temp = pd.DatetimeIndex(dataset.index)
    c_weekday = dataset["Power"][temp.weekday < 5].mean()
    c_weekend = dataset["Power"][temp.weekday >= 5].mean()
    c_max = dataset["Power"].max()
    c_min = dataset["Power"].min()
    c_evening = dataset["Power"].between_time("18:00", "22:00").mean()
    c_morning = dataset["Power"].between_time("6:00", "10:00").mean()
    c_night = dataset["Power"].between_time("1:00", "5:00").mean()
    c_noon = dataset["Power"].between_time("10:00", "14:00").mean()
    # ratios
    mean = dataset["Power"].mean()
    maximum = dataset["Power"].max()
    minimum = dataset["Power"].min()
    r_mean = 0.0
    r_min = 0.0
    r_night = 0.0
    r_morning = 0.0
    r_evening = 0.0
    if (mean > 0 and maximum > 0 and minimum > 0):
        r_mean = mean / maximum
        r_min = minimum / mean
        r_night = c_night / c_day
        r_morning = c_morning / c_noon
        r_evening = c_evening / c_noon
    # temporal properties
    # t_daily_max = dataset["Power"].idxmax()
    s_variance = (dataset["Power"].std()) ** 2
    temp = dataset["Power"].diff()
    temp = temp.fillna(value=0)
    # Statistical properties
    s_diff = sum(temp.values)
    s_num_peaks = sum(temp[temp > 0.2].value_counts())
    s_x_corr = dataset["Power"].autocorr()
    # dataset = dataset.resample('60min', how='median')
    list_of_features = [c_day, c_weekday, c_weekend, c_max, c_min, c_evening, c_morning, c_night, c_noon, r_mean, r_min,
                        r_night, r_morning, r_evening, s_variance, s_diff, s_num_peaks, s_x_corr]
    return list_of_features




def compute(df2):
<<<<<<< HEAD
    df2 = df2[:-1]
    employed = []
    # df2 = get_features(dataset)
    # df2 = df2.fillna(value = 0)
    print "Filling up fp...."
    # fp = get_retirement_status("/Users/Rishi/Downloads/abc.csv")
    # fp = fp[df2["Code"].values]
    # fp = fp.fillna(value = 0)

    fp = df2["Retirement_answers"].values
    for i in range(len(fp)):
        if fp[i] == 6:
            fp[i] = 1
    print "Done!"
    # df2 = df2.drop("Retirement_answers", axis = 1)
    # lis = fp.values
    lis = list(fp)
    df = df2  # .drop("Code", axis = 1)
    print "GETTING HEAD: ", df.head()
    print "GETTING INDEX", df.index
    # df = df.drop("Index", axis = 1)
    df = (df + 0.0)
    print df.head()
    df2_train = df.head(len(df2) / 2)
    print "Training..."
    ground_truth_train = lis[:len(df) / 2]
    df2_test = df.tail(len(df) / 2)
    ground_truth_test = lis[len(df) / 2:]
    dataframe_train_array = np.asarray([i for i in df2_train.values])
    ground_truth_train_array = np.asarray(ground_truth_train)
    dataframe_test_array = np.asarray([i for i in df2_test.values])
    ground_truth_test_array = np.asarray(ground_truth_test)
    print "Done!"
    ground_truth_test_array[ground_truth_test_array == -inf] = 0
    ground_truth_train_array[ground_truth_test_array == -inf] = 0
    dataframe_train_array[dataframe_train_array == -inf] = 0
    dataframe_test_array[dataframe_test_array == -inf] = 0
    ground_truth_test_array[ground_truth_test_array == inf] = 0
    ground_truth_train_array[ground_truth_test_array == inf] = 0
    dataframe_train_array[dataframe_train_array == inf] = 0
    dataframe_test_array[dataframe_test_array == inf] = 0
    print ground_truth_test_array
    print np.isnan(np.min(dataframe_train_array))
    dataframe_train_array = np.nan_to_num(dataframe_train_array)
    ground_truth_train = np.nan_to_num(ground_truth_train)
    dataframe_test_array = np.nan_to_num(dataframe_test_array)
    ground_truth_test = np.nan_to_num(ground_truth_test)
    out = {}
    for clf_name, clf in classifiers_dict.iteritems():
        out[clf_name] = {}
        clf.fit((dataframe_train_array[:100000]), (ground_truth_train_array[:100000]))
        prediction = clf.predict(dataframe_test_array[:100000])
        print "Precision for ", clf_name, " is ", precision_score(ground_truth_test, prediction)
        print "Recall for ", clf_name, " is ", recall_score(ground_truth_test, prediction)
        # print "MCC for ", clf_name, " is ", matthews_corrcoef(ground_truth_test, prediction)
        print "Accuracy for ", clf_name, " is ", accuracy_score(ground_truth_test, prediction)
        for metric_name, metric_func in metric_dict.iteritems():
            out[clf_name][metric_name] = metric_func(ground_truth_test, prediction)
    output_dataframe = pd.DataFrame(out)
    print output_dataframe
=======
	employed = []
	#df2 = get_features(dataset)
	#df2 = df2.fillna(value = 0)
	print "Filling up fp...."
	# fp = get_retirement_status("/Users/Rishi/Downloads/abc.csv")
	# fp = fp[df2["Code"].values]
	# fp = fp.fillna(value = 0)

	fp = df2["Retirement_answers"].values
	for i in range(len(fp)):
		if fp[i] == 5:
			fp[i]=1
	print fp
	print "Done!"
	#df2 = df2.drop("Retirement_answers", axis = 1)
	#lis = fp.values
	lis = list(fp)
	df = df2#.drop("Code", axis = 1)
	print "GETTING HEAD: ", df.head()
	print "GETTING INDEX", df.index
	#df = df.drop("Index", axis = 1)
	df = (df + 0.0)
	print df.head()
	df2_train = df.head(2616)
	print "Training..."
	ground_truth_train = lis[:2616]
	df2_test = df.tail(872)
	ground_truth_test = df2["Retirement_answers"].tail(872)
	ground_truth_test = ground_truth_test.values
	dataframe_train_array = np.asarray([i for i in df2_train.values])
	ground_truth_train_array = np.asarray(ground_truth_train)
	dataframe_test_array = np.asarray([i for i in df2_test.values])
	ground_truth_test_array = np.asarray(ground_truth_test)
	print "Done!"
	ground_truth_test_array[ground_truth_test_array== -inf] = 0
	ground_truth_train_array[ground_truth_test_array== -inf] = 0
	dataframe_train_array[dataframe_train_array== -inf] = 0
	dataframe_test_array[dataframe_test_array == -inf] = 0
	ground_truth_test_array[ground_truth_test_array== inf] = 0
	ground_truth_train_array[ground_truth_test_array== inf] = 0
	dataframe_train_array[dataframe_train_array== inf] = 0
	dataframe_test_array[dataframe_test_array == inf] = 0
	print ground_truth_test_array
	if np.isnan(np.min(dataframe_train_array)):
		dataframe_train_array = np.nan_to_num(dataframe_train_array)
	if np.isnan(np.min(ground_truth_train)):
		ground_truth_train = np.nan_to_num(ground_truth_train)
	if np.isnan(np.min(dataframe_test_array)):
		dataframe_test_array = np.nan_to_num(dataframe_test_array)
	if np.isnan(np.min(ground_truth_test)):
		ground_truth_test = np.nan_to_num(ground_truth_test)
	out = {}
	for clf_name, clf in classifiers_dict.iteritems():
		out[clf_name] = {}
		clf.fit((dataframe_train_array),(ground_truth_train_array))
		prediction = clf.predict(dataframe_test_array)
		print "Precision for ", clf_name, " is ", precision_score(ground_truth_test, prediction)
		print "Recall for ", clf_name, " is ", recall_score(ground_truth_test, prediction)
		#print "MCC for ", clf_name, " is ", matthews_corrcoef(ground_truth_test, prediction)
		print "Accuracy for ", clf_name, " is ", accuracy_score(ground_truth_test, prediction)
		for metric_name, metric_func in metric_dict.iteritems():
			out[clf_name][metric_name] = metric_func(ground_truth_test, prediction)
	output_dataframe = pd.DataFrame(out)
	print output_dataframe
>>>>>>> 468256f9912089df156d7aea850bd85b52ea699a


def get_csv_ground_truth(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df


def get_number_of_bedrooms(path_to_csv):
<<<<<<< HEAD
    df = get_csv_ground_truth(path_to_csv)
    dataset = df["Question 460: How many bedrooms are there in your home"].fillna(value=0)
    return dataset

=======
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 460: How many bedrooms are there in your home"].fillna(value = 0)
	temp_arr = []
	for i in dataset.values:
		if i < 2:
			temp_arr.append(0)
		elif i==3:
			temp_arr.append(1)
		elif i==4:
			temp_arr.append(2)
		else:
			temp_arr.append(3)
	df = pd.DataFrame(temp_arr, index = dataset.index, columns = ["Question 460: How many bedrooms are there in your home"])
	return df
>>>>>>> 468256f9912089df156d7aea850bd85b52ea699a

def get_type_of_cooking(path_to_csv):
    df = get_csv_ground_truth(path_to_csv)
    dataset = df["Question 4704: Which of the following best describes how you cook in your home"].fillna(value=0)
    return dataset


def get_employment_status(path_to_csv):
    df = get_csv_ground_truth(path_to_csv)
    dataset = df["Question 310: What is the employment status of the chief income earner in your household, is he/she"][
        df["Question 310: What is the employment status of the chief income earner in your household, is he/she"] < 4]
    dataset = dataset.fillna(value=0)
    return dataset


def get_retirement_status(path_to_csv):
<<<<<<< HEAD
    df = get_csv_ground_truth(path_to_csv)
    dataset = df["Question 310: What is the employment status of the chief income earner in your household, is he/she"][
        df["Question 310: What is the employment status of the chief income earner in your household, is he/she"] == 6]
    dataset = dataset.fillna(value=0)
    return dataset

=======
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 310: What is the employment status of the chief income earner in your household, is he/she"]#[df["Question 310: What is the employment status of the chief income earner in your household, is he/she"] == 6]
	#dataset = dataset.fillna(value = 0)
	dataset_array = dataset.values
	for i in range(len(dataset_array)):
		if (int(dataset_array[i]) == 6):
			dataset_array[i] = 1.0
		else:
			dataset_array[i] = 0.0
	return pd.DataFrame(dataset_array, index = dataset.index)
>>>>>>> 468256f9912089df156d7aea850bd85b52ea699a

def get_family(path_to_csv):
    df = get_csv_ground_truth(path_to_csv)
    dataset = df["Question 410: What best describes the people you live with? READ OUT"][
        df["Question 410: What best describes the people you live with? READ OUT"] > 1]
    dataset = dataset.fillna(value=0)
    return dataset

dataset_path = os.path.expanduser("~/CER")
csv_path = os.path.expanduser("~/CER/CER_both/electricity/csv/pre_trial.csv")

<<<<<<< HEAD
#edf = get_dataset_concat(dataset_path)
#fm = create_feature_matrix(edf, csv_path)
=======
def get_children(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 43111: How many people under 15 years of age live in your home?"][df["Question 43111: How many people under 15 years of age live in your home?"]>=1]
	dataset.replace(' ', np.nan, inplace = True)
	dataset = dataset.fillna(value= 0)
	#[df["Question 43111: How many people under 15 years of age live in your home?"]> 1]
	#dataset = pd.DataFrame(dataset)
	dataset = dataset.fillna(value = 0)
	print dataset.index
	return dataset

def get_single(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	dataset = pd.DataFrame(df["Question 410: What best describes the people you live with? READ OUT"])
	print type(dataset)
	dataset = dataset["Question 410: What best describes the people you live with? READ OUT"]
	lis = []
	for i in dataset.values:
		if i==1:
			lis.append(1)
		else:
			lis.append(0)
	dataset = pd.DataFrame(lis, index = dataset.index, columns = ["Question 410: What best describes the people you live with? READ OUT"])
	print dataset
	# def get_age_house(path_to_csv):

def get_adults(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 420: How many people over 15 years of age live in your home?"][df["Question 420: How many people over 15 years of age live in your home?"]>=1]
	dataset.replace(' ', np.nan, inplace = True)
	dataset = dataset.fillna(value= 0)
	#[df["Question 43111: How many people under 15 years of age live in your home?"]> 1]
	#dataset = pd.DataFrame(dataset)
	dataset = dataset.fillna(value = 0)
	print dataset.index
	return dataset

def get_number_of_residents(path_to_csv):
	dataset_children = get_children(path_to_csv)
	dataset_adults = get_adults(path_to_csv)
	temp_arr = [None] * len(dataset_adults)
	int_arr = [None] * len(dataset_adults)
	for i in range(len(dataset_children)):
		temp = int(dataset_children[i]) + int(dataset_adults[i])
		if (temp<=2 and temp!=0):
			temp_arr[i] = "Few"
			int_arr[i] = 0
		else:
			temp_arr[i] = "Many"
			int_arr[i] = 1
	dataset_residents = pd.DataFrame(int_arr, index = dataset_children.index, columns = ["Number of residents"])
	return dataset_residents

def get_age_house(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	dataset = df["Question 4531: Approximately how old is your home?"]
	dataset.replace(' ', np.nan, inplace = True)
	dataset = dataset.fillna(value= 0)
	print dataset
	temp_arr = []
	int_arr = []
	print len(dataset.values)
	for i in range(len(dataset.values)):
		if int(dataset.values[i]) < 4:
			int_arr.append(0)
			temp_arr.append(0)
		else:
			temp_arr.append(1)
			int_arr.append(1)
	return pd.DataFrame(int_arr, index = dataset.index, columns = ["Question 4531: Approximately how old is your home?"])

def get_social_class(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	df = df["Question 401: SOCIAL CLASS Interviewer, Respondent said that occupation of chief income earner was.... <CLASS> Please code"]
	df_values = df.values
	temp_arr = []
	int_arr = []
	for i in df_values:
		print i, type(i)
		if int(i)==1:
			int_arr.append(0)
			temp_arr.append('AB')
		elif int(i)==2 or int(i)==3:
			int_arr.append(1)
			temp_arr.append('C1C2')
		elif int(i)==4:
			int_arr.append(2)
			temp_arr.append('DE')
		else:
			int_arr.append(3)
			temp_arr.append("")
	return pd.DataFrame(int_arr, index=df.index, columns = ["social_class"])

def get_floor_area(path_to_csv):
	df = get_csv_ground_truth(path_to_csv)
	floor_area_df = df["Question 6103: What is the approximate floor area of your home?"]
	unit_df = df["Question 61031: Is that"]
	unit_df.replace(' ', 0, inplace = True)
	unit_df.replace('1', 1, inplace = True)
	unit_df.replace('2', 2, inplace = True)
	floor_area_array = floor_area_df.values
	unit_array = unit_df.values
	int_arr = []
	for i in range(len(floor_area_array)):
		if (unit_array[i] == 2):
			floor_area_array[i] = float(floor_area_array[i])/10.91
		elif (int(unit_array[i]) == 1):
			floor_area_array[i] = float(floor_area_array[i])
		else:
			floor_area_array[i] = -1
	for i in range(len(floor_area_array)):
		if floor_area_array[i] < 100:
			int_arr.append(0)
		elif floor_area_array[i] >=100 and floor_area_array[i] <= 200:
			int_arr.append(1)
		elif floor_area_array[i] > 200:
			int_arr.append(2)
		else:
			int_arr.append(-1)
	return pd.DataFrame(int_arr, index=df.index, columns = ["floor_area"])
def get_number_of_appliances(path_to_csv):
	list_of_questions = ["Question 49001: Washing machine",	"Question 49001: Tumble dryer", "Question 49001: Dishwasher", "Question 49001: Electric shower (instant)", "Question 49001: Electric shower (electric pumped from hot tank)", "Question 49001: Electric cooker",	"Question 49001: Electric heater (plug-in convector heaters)",	"Question 49001: Stand alone freezer",	"Question 49001: A water pump or electric well pump or pressurised water system",	"Question 49001: Immersion"	, "Question 49002: Washing machine",	"Question 49002: Tumble dryer"	,"Question 49002: Dishwasher",	"Question 49002: Electric shower (instant)", "Question 49002: Electric shower (electric pumped from hot tank)", "Question 49002: Electric cooker",	"Question 49002: Electric heater (plug-in convector heaters)",	"Question 49002: Stand alone freezer",	"Question 49002: A water pump or electric well pump or pressurised water system",	"Question 49002: Immersion",	"Question 4902: TV's less  than 21 inch",	"Question 4902: TV's greater  than 21 inch", "Question 4902: Desk-top computers",	"Question 4902: Lap-top computers", 	"Question 4902: Games consoles, such as xbox, playstation or Wii",	"Question 490002: TV's less than 21 inch",	"Question 490002: TV's greater than 21 inch"	"Question 490002: Desk-top computers",	"Question 490002: Lap-top computers",	"Question 490002: Games consoles, such as xbox, playstation or Wii",	"Question 49004: Washing machine  INT:PROBE TO PRECODES",	"Question 49004: Tumble dryer  INT:PROBE TO PRECODES",	"Question 49004: Dishwasher  INT:PROBE TO PRECODES",	"Question 490004: Electric shower (instant)  INT:PROBE TO PRECODES",	"Question 4900004: Electric shower (pumped from hot tank)  INT:PROBE TO PRECODES",	"Question 4900005: Electric cooker  INT:PROBE TO PRECODES",	"Question 4900006: Electric heater (plug-in)  INT:PROBE TO PRECODES", "Question 4900007: Water pump   INT:PROBE TO PRECODES",	"Question 4900008: Immersion water  INT:PROBE TO PRECODES",	"Question 4900009: Stand alone Freezer  INT:PROBE TO PRECODES",	"Question 49022: TV’s less than 21 inch  INT:PROBE TO PRECODES", "Question 49022: TV’s greater than 21 inch  INT:PROBE TO PRECODES", "Question 49022: Desk-top computers  INT:PROBE TO PRECODES", "Question 49022: Lap-top computers  INT:PROBE TO PRECODES", "Question 49022: Games consoles, such as xbox, playstation or Wii  INT:PROBE TO PRECODES"]
#print get_floor_area("/Users/Rishi/Downloads/abc.csv")
path_to_txt = "/Users/Rishi/Downloads"
get_dataset(path_to_txt)
>>>>>>> 468256f9912089df156d7aea850bd85b52ea699a
