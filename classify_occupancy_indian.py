from os import *
from os.path import *
from itertools import chain
import os
from sklearn import tree
import pandas as pd
from numpy import *
from pandas import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pytz
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from common_functions import  *
import itertools
import numpy as np
np.random.seed(42)

geyser_power = 2200
fridge_power = 150
results_dic = {}

classifiers_dict = {"SVM": svm.SVC(), "KNN": KNeighborsClassifier(n_neighbors=3),
                    "DT": tree.DecisionTreeClassifier(),"RF":RandomForestClassifier()}



df_power= pd.read_csv("/Users/nipunbatra/git/energylensplus-offline/energylenserver/data/meter/103/PowerMeter.csv", index_col=0)
df_light= pd.read_csv("/Users/nipunbatra/git/energylensplus-offline/energylenserver/data/meter/103/LightMeter.csv", index_col=0)

df_power.index = pd.to_datetime(df_power.index, unit='s')
df_power_copy = df_power.copy()
df_power = df_power.resample("1T")

df_light.index = pd.to_datetime(df_light.index, unit='s')
df_light_copy = df_light.copy()
df_light = df_light.resample("1T")


df = df_power+df_light
#df = df["2015-02-01"]

d = pd.read_csv("/Users/nipunbatra/git/energylensplus-offline/ground_truth/103_353325066658901_presencelog.csv", index_col=0)
d.index = pd.to_datetime(d.index, unit='s')
d['location']=1
d = d.resample('1T')
d=d.fillna(0)
df["occupancy"]=d


def tp_score(gt, pred):
    gt = gt.copy().dropna()
    pred = pred.copy().dropna()
    index_intersection = gt.index.intersection(pred.index)
    gt_x = gt.ix[index_intersection]
    pred_x = pred.ix[index_intersection]
    tp = ((gt_x==1)&(pred_x==1)).sum()*1.0/((gt_x==1).sum())
    return tp

def tn_score(gt, pred):
    gt = gt.copy().dropna()
    pred = pred.copy().dropna()
    index_intersection = gt.index.intersection(pred.index)
    gt_x = gt.ix[index_intersection]
    pred_x = pred.ix[index_intersection]
    tn = ((gt_x==0)&(pred_x==0)).sum()*1.0/((gt_x==0).sum())
    return tn


def fp_score(gt, pred):
    gt = gt.copy().dropna()
    pred = pred.copy().dropna()
    index_intersection = gt.index.intersection(pred.index)
    gt_x = gt.ix[index_intersection]
    pred_x = pred.ix[index_intersection]
    fp = ((gt_x==0)&(pred_x==1)).sum()*1.0/((gt_x==0).sum())
    return fp


def fn_score(gt, pred):
    gt = gt.copy().dropna()
    pred = pred.copy().dropna()
    index_intersection = gt.index.intersection(pred.index)
    gt_x = gt.ix[index_intersection]
    pred_x = pred.ix[index_intersection]
    fn= ((gt_x==1)&(pred_x==0)).sum()*1.0/((gt_x==1).sum())
    return fn

metric_dict = {"Precision":precision_score, "Recall":recall_score,
               "MCC":matthews_corrcoef, "Accuracy":accuracy_score,
               "tp":tp_score, "fp":fp_score, "tn":tn_score, "fn":fn_score}

def accuracy_metrics(test_gt, prediction):
    tp=None
    tn=None
    fp=None
    fn=None
    return fp

def classify_train_test_split(folder_path):
    out = {"summer":{}, "winter":{}}
    X_test_dict = {"summer":{}, "winter":{}}
    y_test_dict = {"summer":{}, "winter":{}}
    X_train_dict = {"summer":[], "winter":[]}
    y_train_dict = {"summer":[], "winter":[]}
    y_train_list = {"summer":{}, "winter":{}}
    X_train_list = {"summer":{}, "winter":{}}


    for season in ["summer", "winter"]:
        season_path = os.path.join(folder_path, season)
        for home in [1, 2, 3, 4, 5]:
            home_path = season_path+"/"+str(home)+".csv"
            df = pd.read_csv(home_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.between_time("06:00", "22:00")
            X = df
            y = X.pop('occupancy')
            X_train_idx,X_test_idx,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
            X_train_dict[season].append(X.ix[X_train_idx])
            X_test_dict[season][home]=X.ix[X_test_idx]
            y_train_dict[season].append(y_train)
            y_test_dict[season][home]=y_test
        y_train_list[season]=list(itertools.chain(*y_train_dict[season]))
        X_train_list[season] = pd.concat(X_train_dict[season])

    for season in ["summer", "winter"]:
        for home in [1, 2, 3, 4, 5]:
            out[season][home] = {}
            for clf_name, clf in classifiers_dict.iteritems():
                out[season][home][clf_name] = {}
                clf.fit(X_train_list[season], y_train_list[season])
                pred = clf.predict(X_test_dict[season][home])
                pred = pd.Series(pred, index=X_test_dict[season][home].index)
                gt = pd.Series(y_test_dict[season][home], index=X_test_dict[season][home].index)
                for metric_name, metric_func in metric_dict.iteritems():
                    out[season][home][clf_name][metric_name] = metric_func(gt, pred)

    return out

a = None

def our_approach(df):
    df = df.between_time("06:00", "22:00").copy()
    from nilmtk.feature_detectors.steady_states import find_steady_states, find_steady_states_transients
    from nilmtk.disaggregate.hart_85 import Hart85
    h = Hart85()
    geyser_min = geyser_power - 200
    geyser_max = geyser_power + 200
    fridge_min = fridge_power-40
    fridge_max = fridge_power+40

    pred_index = pd.DatetimeIndex(start="2015-02-01", end="2015-02-13", freq='1s')
    pred = pd.DataFrame({"occupancy":np.zeros(len(pred_index))}, index=pred_index)

    # Look at power stream
    ss_power, tr_power = find_steady_states(df_power_copy)
    days_power = pd.DatetimeIndex(np.unique(tr_power.index.date))

    ss_light, tr_light = find_steady_states(df_light_copy)
    days_light = pd.DatetimeIndex(np.unique(tr_light.index.date))

    days = pd.DatetimeIndex(np.union1d(days_light, days_power))

    # Find last non-fridge and non-geyser event
    for day in days:
        light_exists = False
        try:
            tr_day_light = tr_light[day.strftime("%Y-%m-%d")]
            light_exists = True
        except:
            light_exists=False
            pass

        tr_day_power = tr_power[day.strftime("%Y-%m-%d")].abs()

        # Check for transitions which are not due to geyser or fridge
        tr_day_power_non_bg = tr_day_power[(tr_day_power['active transition']<fridge_min) | (tr_day_power['active transition']>fridge_max)
        |(tr_day_power['active transition']<geyser_min)|(tr_day_power['active transition']>geyser_max)]

        if len(tr_day_power_non_bg)>0:
            # Some other appliance exists
            last_time_power = tr_day_power_non_bg.index[-1]

            if light_exists and len(tr_day_light)>0:
                last_time_light = tr_day_light.index[-1]

                if last_time_light<=last_time_power:
                    pred[day.strftime("%Y-%m-%d")][last_time_power:]=1
                else:
                    pred[day.strftime("%Y-%m-%d")][last_time_light:]=1
            else:
                pred[day.strftime("%Y-%m-%d")][last_time_power:]=1

        else:
            if light_exists and len(tr_day_light)>0:
                last_time_light = tr_day_light.index[-1]
                pred[day.strftime("%Y-%m-%d")][last_time_light:]=1


    from copy import deepcopy
    # Pair everything in light and signal occupancy between them
    h.transients = deepcopy(tr_light)

    pair_df_light = h.pair(buffer_size=20,
              min_tolerance=100, percent_tolerance=0.035,
              large_transition=1000)

    for idx, row in pair_df_light.iterrows():
        start = row['T1 Time']
        end = row['T2 Time']
        pred[start:end] = 1


    # Pair non-bg loads
    tr_non_bg_power_overall = tr_power[(tr_power['active transition']<fridge_min) | (tr_power['active transition']>fridge_max)
        |(tr_power['active transition']<geyser_min)|(tr_power['active transition']>geyser_max)]
    h.transients = deepcopy(tr_non_bg_power_overall)
    pair_df_power = h.pair(buffer_size=20,
              min_tolerance=100, percent_tolerance=0.035,
              large_transition=1000)

    for idx, row in pair_df_light.iterrows():
        start = row['T1 Time']
        end = row['T2 Time']
        pred[start:end] = 1


    pred.ix[tr_light.index] = 1
    #pred.ix[tr_non_bg_power_overall.index] = 1


    return pred.resample("15T", how="max")






def nilm_pair_remove_fridge(df, start_event_remove=True):
    global  a
    from copy import  deepcopy
    df = df.between_time("06:00", "22:00").copy()
    from nilmtk.feature_detectors.steady_states import find_steady_states, find_steady_states_transients
    from nilmtk.disaggregate.hart_85 import Hart85
    h = Hart85()

    geyser_min = geyser_power - 100
    geyser_max = geyser_power + 100
    fridge_min = fridge_power-100
    fridge_max = fridge_power+100

    # Train on DF to get pairs.
    # 1. Between pairs put occupancy as 1
    # 2. If not a pair, then also put occupancy as 1
    pred_index = pd.DatetimeIndex(start="2015-02-01", end="2015-02-13", freq='1s')
    pred = pd.DataFrame({"occupancy":np.zeros(len(pred_index))}, index=pred_index)
    #pred = pd.DataFrame({"occupancy":np.zeros(len(df_power_copy))}, index=df_power_copy.index)

    ss, tr = find_steady_states(df_power_copy)
    days = pd.DatetimeIndex(np.unique(tr.index.date))


    #for df_data in [df_power_copy]:
    for num, df_data in enumerate([df_light_copy, df_power_copy]):
        ss, tr = find_steady_states(df_data)
        days = pd.DatetimeIndex(np.unique(tr.index.date))

        # Find unique days

        for day in days:
            events_day = tr[day.strftime("%Y-%m-%d")].abs()
            # Find first non-fridge event
            event_df = events_day
            #event_df = events_day[(events_day['active transition']<=fridge_min)|(events_day['active transition']>=fridge_max)]
            #event_df = event_df[(event_df['active transition']<=geyser_min)|(events_day['active transition']>=geyser_max)]
            if len(event_df)>0:
                first_event=event_df.index[0]
                last_event = event_df.index[-1]
                if start_event_remove:
                    pred[day.strftime("%Y-%m-%d")][:first_event]=1
                pred[day.strftime("%Y-%m-%d")][last_event:]=1

        for ix, row in tr.iterrows():

            if num==0:
                #Light, every thing should be added to TP!
                pred.ix[tr.index] = 1
            else:
                # Power, ignore fridge and heater
                if not(fridge_min<=row['active transition']<=fridge_max or geyser_min<=row['active transition']<=geyser_max):
                    print row, "A"
                    pred.ix[tr.index] = 1

        h.transients = deepcopy(tr)

        pair_df = h.pair(buffer_size=20,
                  min_tolerance=100, percent_tolerance=0.035,
                  large_transition=1000)
        for idx, row in pair_df.iterrows():
            start = row['T1 Time']
            end = row['T2 Time']
            if num==0:
                pred[start:end] = 1
            else:
                if not((fridge_min<=row['T1 Active']<=fridge_max) or (geyser_min<=row['T1 Active']<=geyser_max)):
                    print "-----Filling between----", row

                    pred[start:end] = 1
                else:
                    print "********Fridge or geyser**********", row



    pred_resampled = pred.resample("15T", how="max")
    return pred_resampled
    #return pred_resampled
    print pred_resampled.sum(), "Our"
    gt_occupancy = df["occupancy"].resample("15T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred_resampled.index)
    return {"tp":tp_score(gt_occupancy, pred_resampled),
                "fp":fp_score(gt_occupancy, pred_resampled),
                "tn":tn_score(gt_occupancy, pred_resampled),
                "fn":fn_score(gt_occupancy, pred_resampled),
            "Accuracy":
            accuracy_score(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection]),
            "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection])
            }

def nilm_pair(df,freq="15", flag=0):
    from copy import deepcopy
    from nilmtk.feature_detectors.steady_states import find_steady_states, find_steady_states_transients
    from nilmtk.disaggregate.hart_85 import Hart85
    h = Hart85()

    # Train on DF to get pairs.
    # 1. Between pairs put occupancy as 1
    # 2. If not a pair, then also put occupancy as 1
    ss, tr = find_steady_states(df[["power"]])
    pred = pd.DataFrame({"occupancy":np.zeros(len(df))}, index=df.index)

    pred.ix[tr.index] = 1
    h.transients = deepcopy(tr)

    pair_df = h.pair(buffer_size=20,
              min_tolerance=100, percent_tolerance=0.035,
              large_transition=1000)
    for idx, row in pair_df.iterrows():
        start = row['T1 Time']
        end = row['T2 Time']
        pred[start:end] = 1

    pred_resampled = pred.resample(str(freq)+"T", how="max")
    return pred_resampled
    print pred_resampled.sum(), "Pair"
    gt_occupancy = df["occupancy"].resample(str(freq)+"T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred_resampled.index)
    if flag is 0:
        return {"tp":tp_score(gt_occupancy, pred_resampled),
                "fp":fp_score(gt_occupancy, pred_resampled),
                "tn":tn_score(gt_occupancy, pred_resampled),
                "fn":fn_score(gt_occupancy, pred_resampled),
                "Accuracy":
                accuracy_score(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection]),
                "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection])
                }
    else:
        return {}


def chen(df, how="max"):
    print (len(df)), len(df.between_time("01:00", "04:00"))
    pred_index = df.between_time("06:00", "22:00").resample("15T").index
    pred = pd.DataFrame({"occupancy":np.zeros(len(pred_index))},
                        index=pred_index)

    days = pd.DatetimeIndex(np.unique(df.index.date))
    print days
    for day in days:

        night_df = df[day.strftime("%Y-%m-%d")].between_time("01:00", "04:00")["power"]
        print ("Length", len(night_df)), len(df[day.strftime("%Y-%m-%d")]), len(df)
        if len(night_df):
            print "First"
            if how is "median":
                avg_power_threshold = night_df.resample("15T").mean()
                avg_range_threshold = (night_df.resample("15T", how="max")-night_df.resample("15T", how="min")).mean()
                avg_sd_threshold = night_df.resample("15T", how="std").mean()
            else:
                avg_power_threshold = night_df.resample("15T").max()
                avg_range_threshold = (night_df.resample("15T", how="max")-night_df.resample("15T", how="min")).max()
                avg_sd_threshold = night_df.resample("15T", how="std").max()
        else:
            print "Second"
            if how is "median":
                avg_power_threshold = df["power"].mean()
                avg_range_threshold = df["power"].mean()
                avg_sd_threshold = df["power"].mean()
            else:
                avg_power_threshold = df["power"].max()
                avg_range_threshold = df["power"].max()
                avg_sd_threshold = df["power"].max()
        print avg_power_threshold, avg_range_threshold, avg_sd_threshold
        day_df = df.between_time("06:00", "22:00")["power"]
        day_df_resampled_power = day_df.resample("15T")
        day_df_resampled_range = day_df.resample("15T", how="max")-day_df.resample("15T", how="min")
        day_df_resampled_sd = day_df.resample("15T", how="std")
        try:
            pred.ix[day_df_resampled_power.index] = (day_df_resampled_power>avg_power_threshold) |\
                                                (day_df_resampled_range>avg_range_threshold) |(day_df_resampled_sd>avg_sd_threshold)
        except:
            pass

    pred_downsample = pred.resample("60T", how="max")
    pred = pred_downsample.resample("15T").fillna(method='ffill')
    return pred
    print pred.sum(), "Chen"
    gt_occupancy = df["occupancy"].resample("15T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred.index)
    return {"tp":tp_score(gt_occupancy, pred),
                "fp":fp_score(gt_occupancy, pred),
                "tn":tn_score(gt_occupancy, pred),
                "fn":fn_score(gt_occupancy, pred),"Accuracy":
            accuracy_score(gt_occupancy.ix[index_intersection], pred.ix[index_intersection]),
            "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred.ix[index_intersection])
            }


def nilm_naive(df):

    from nilmtk.feature_detectors.steady_states import find_steady_states, find_steady_states_transients
    """
    X = df
    y = X.pop('occupancy')
    X_train_idx,X_test_idx,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
    X_train = X.ix[X_train_idx]
    X_test = X.ix[X_test_idx]
    """
    ss, tr = find_steady_states(df[["power"]])
    pred = pd.DataFrame({"occupancy":np.zeros(len(df))}, index=df.index)

    pred.ix[tr.index] = 1
    pred_resampled = pred.resample("15T", how="max")
    return pred_resampled
    gt_occupancy = df["occupancy"].resample("15T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred_resampled.index)
    return {"tp":tp_score(gt_occupancy, pred_resampled),
                "fp":fp_score(gt_occupancy, pred_resampled),
                "tn":tn_score(gt_occupancy, pred_resampled),
                "fn":fn_score(gt_occupancy, pred_resampled),
            "Accuracy":
            accuracy_score(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection]),
            "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection])
            }


def sensitivity_our_approach(folder_path):
    out = {}
    for season in ["summer", "winter"]:
        out[season] = {}
        season_path = os.path.join(folder_path, season)
        for home in [1, 2, 3, 4, 5]:
        #for home in  [4]:
            print home, season
            print "*"*80
            out[season][home] = {}
            home_path = season_path+"/"+str(home)+".csv"
            df = pd.read_csv(home_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.between_time("06:00", "22:00").copy()
            for freq in [5, 10, 15, 20, 25]:
                out[season][home][freq] = nilm_pair(home, season,
                                                                   os.path.expanduser("~/Dropbox/niomtk_data/eco/1min/"),
                                                    freq)
    return out

def classify_train_test_same_home(df):
    out = {}

    df = df.dropna()
    df = df.between_time("06:00", "22:00")
    X = df.copy()
    y = X.pop('occupancy')
    train_size = len(df)/2
    X_train,X_test,y_train,y_test = X.head(train_size), X.tail(train_size), y.head(train_size), y.tail(train_size)
    df_test = df.tail(train_size)
    for clf_name, clf in classifiers_dict.iteritems():
        out[clf_name] = {}
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred = pd.Series(pred, index=X_test.index)
        out[clf_name] = pred.resample("15T", how="max")
    out["gt"] = pd.Series(y_test, index=X_test.index).resample("15T", how="max")

    #out["NILM naive"] = nilm_naive(df_test)

    out["Chen"] = chen(df_test)
    out["Chen-median"] = chen(df_test, how="median")
    #out["NILM pairing"] = nilm_pair(df_test)
    out["Our approach"] = nilm_pair_remove_fridge(df_test)
    out["Our approach-optimised"] = nilm_pair_remove_fridge(df_test,start_event_remove=False )
    #out["New"] = our_approach(df_test)
    #out["Always occupied"] = pd.Series(np.ones(len(y_test)), index=X_test.index)

    return out





def plot_train_single(metric_plot="Accuracy"):
    out = classify_train_test_same_home(output_path)
    res = {}
    for season, season_dict in out.iteritems():
        fig, ax = plt.subplots(nrows=3)
        res[season] = {}
        for metric in metric_dict.keys():
            res[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                temp = pd.DataFrame(home_results).T.to_dict()
                if metric in temp:
                    res[season][metric][home] = temp[metric]
        # Plotting
    latexify(columns=2, fig_height=4)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    summer_df = pd.DataFrame(res['summer'][metric_plot]).T
    winter_df = pd.DataFrame(res['winter'][metric_plot]).T

    summer_df.plot(ax=ax[0], rot=0, title="Summer", kind="bar").legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.4))
    winter_df.plot(ax=ax[1], rot=0, title="Winter", kind="bar", legend=False)

    format_axes(ax[0])
    format_axes(ax[1])

    ax[1].set_xlabel("Household")
    ax[0].set_ylabel(metric_plot)
    ax[1].set_ylabel(metric_plot)


    plt.tight_layout()
    plt.savefig("figures/same_home_train_%s.pdf" %metric_plot, bbox_inches="tight")
    plt.savefig("figures/same_home_train_%s.png" %metric_plot, bbox_inches="tight")

def plot_train_all():
    out = classify_train_test_split(output_path)
    res = {}
    fig, ax = plt.subplots(nrows=3)
    for season, season_dict in out.iteritems():

        res[season] = {}
        for metric in metric_dict.keys():
            res[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                res[season][metric][home] = pd.DataFrame(home_results).T.to_dict()[metric]
    # Plotting
    latexify(fig_height=3.2)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    summer_df = pd.DataFrame(res['summer']['Accuracy']).T
    winter_df = pd.DataFrame(res['winter']['Accuracy']).T

    summer_df.plot(ax=ax[0], rot=0, title="Summer", kind="bar").legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    winter_df.plot(ax=ax[1], rot=0, title="Winter", kind="bar", legend=False)

    format_axes(ax[0])
    format_axes(ax[1])

    ax[1].set_xlabel("Household")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")


    #plt.tight_layout()
    plt.savefig("figures/split.pdf", bbox_inches="tight")

def plot_train_all_drop():
    out1 = classify_train_test_same_home(output_path)
    out2 = classify_train_test_split(output_path)
    res1 = {}
    res2= {}
    for season, season_dict in out1.iteritems():

        res1[season] = {}
        for metric in metric_dict.keys():
            res1[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                res1[season][metric][home] = pd.DataFrame(home_results).T.to_dict()[metric]
    for season, season_dict in out2.iteritems():

        res2[season] = {}
        for metric in metric_dict.keys():
            res2[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                res2[season][metric][home] = pd.DataFrame(home_results).T.to_dict()[metric]
        # Plotting
    latexify(fig_height=2.6)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    summer_df_1 = pd.DataFrame(res1['summer']['Accuracy']).T
    winter_df_1 = pd.DataFrame(res1['winter']['Accuracy']).T

    summer_df_2 = pd.DataFrame(res2['summer']['Accuracy']).T
    winter_df_2 = pd.DataFrame(res2['winter']['Accuracy']).T

    (summer_df_1-summer_df_2).plot(ax=ax[0], rot=0, title="Summer", kind="bar").legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    (winter_df_1-winter_df_2).plot(ax=ax[1], rot=0, title="Winter", kind="bar", legend=False)

    format_axes(ax[0])
    format_axes(ax[1])

    ax[1].set_xlabel("Household")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")


    #plt.tight_layout()
    plt.savefig("figures/split.pdf", bbox_inches="tight")

def compute_metrics(gt, pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    gt = gt.copy().dropna()
    pred = pred.copy().dropna()
    pred = pred.squeeze()
    index_intersection = gt.index.intersection(pred.index)
    gt_x = gt.ix[index_intersection].between_time("06:00", "22:00")
    pred_x = pred.ix[index_intersection].between_time("06:00", "22:00")


    tp = ((gt_x==1)&(pred_x==1)).sum()*1.0/((gt_x==1).sum())
    tn = ((gt_x==0)&(pred_x==0)).sum()*1.0/((gt_x==0).sum())
    fp = ((gt_x==0)&(pred_x==1)).sum()*1.0/((gt_x==0).sum())
    fn= ((gt_x==1)&(pred_x==0)).sum()*1.0/((gt_x==1).sum())
    accuracy = (gt_x==pred_x).sum()*1.0/len(gt_x)
    mcc = matthews_corrcoef(y_pred=pred_x, y_true=gt_x)
    return {"tp":tp, "tn":tn, "fp":fp, "fn":fn, "accuracy":accuracy,"MCC":mcc}



out = classify_train_test_same_home(df)
results = {}
for approach, pred in out.iteritems():
    print approach, type(approach)
    if approach!="gt":
        results[approach] = compute_metrics(out['gt'], pred)

latexify(columns=2, fig_height=2.8)
#results_df=pd.DataFrame(results)[["Chen", "Chen-median","Our approach", "Our approach-optimised", "DT", "SVM", "RF"]]
results_df=pd.DataFrame(results)[["Chen", "Chen-median","Our approach-optimised", "DT", "SVM", "RF"]]
results_df = results_df.rename(columns={"Our approach-optimised":"Our approach"})
results_df = results_df.ix[['accuracy','fn','fp','tn','tp']]
results_df.index=["Accuracy","FN","FP","TN","TP"]
ax = results_df.plot(rot=0, kind="bar")
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
format_axes(ax)
plt.savefig(os.path.expanduser("~/git/NIOMTK/figures/iawe.pdf"),bbox_inches="tight")
plt.clf()
import seaborn as sns
sns.reset_orig()
latexify(columns=1, fig_height=1.6)
results_df.loc['TP+FP'] = results_df.loc['TP'] + results_df.loc['FP']
y = results_df.ix['TP+FP']
x = results_df.ix['FN']

for i,(x_p, y_p) in enumerate(zip(x, y)):
    plt.plot(x_p, y_p, marker='o', linestyle='', label=x.index[i],ms=6)
plt.legend(numpoints=1, loc='upper right')
plt.xlabel("Miss time (FN)")
plt.ylabel("Energy consumption (TP+FP)")
plt.ylim((-0.1,2.0))
plt.xlim((0,1.1))
format_axes(plt.gca())
plt.savefig(os.path.expanduser("~/git/NIOMTK/figures/iawe_scatter.pdf"),bbox_inches="tight")

"""
plt.clf()
sns.reset_orig()
latexify()
fig, ax = plt.subplots(nrows=2, sharex=True)
df["2015-02-01"]["power"].plot(ax=ax[0])
df["2015-02-01"]["occupancy"].plot(ax=ax[1], kind="area",alpha=0.3)
ax[1].set_ylim((-0.2, 1.2))
ax[1].set_xlabel("Time")
format_axes(ax[0])
format_axes(ax[1])
ax[1].set_ylabel("Occupancy")
ax[0].set_ylabel("Power (W)")
plt.tight_layout()
plt.savefig(os.path.expanduser("~/git/NIOMTK/figures/occupancy.pdf"),bbox_inches="tight")


plt.clf()
ax = df["2015-02-01 10:10":"2015-02-01 23:59"]["power"].plot(color='black', linewidth=0.7)
ax.axvspan(pd.to_datetime("2015-02-01 10:00"), pd.to_datetime("2015-02-01 12:00"),facecolor='green',edgecolor='green',alpha=0.1)
ax.axvspan(pd.to_datetime("2015-02-01 16:30"), pd.to_datetime("2015-02-01 23:59"),facecolor='green',edgecolor='green',alpha=0.1)

#df["2015-02-01 10:10":"2015-02-01 23:59"]["occupancy"].plot(secondary_y=True, kind="area",alpha=0.1, color="red")
format_axes(ax)
ax.set_ylabel("Power (W)")
ax.set_xlabel("Time")
#format_axes(ax.twinx())
plt.savefig(os.path.expanduser("~/git/NIOMTK/figures/occupancy.pdf"),bbox_inches="tight")




"""

"""

latexify()
for metric in ["MCC", "fn","fp"]:
    fig, ax = plt.subplots()
    pd.DataFrame(results).ix[metric].plot(ax= ax, kind='bar', rot=0).legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    plt.xlim((0,1))
    plt.xlabel("Metric")
    format_axes(ax)
    plt.savefig(os.path.expanduser("~/git/NIOMTK/figures/single_%s.pdf" %metric),bbox_inches="tight")
"""