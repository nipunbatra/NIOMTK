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


def tp(gt, pred):
    return np.sum((gt==1) &(pred==1))/1.0*np.size(gt)

fridge_power = {1:48,
                2:74,
                3:64,
                4:94,
                5:113}

gt_power = {'summer':{},'winter':{}}
pair_df_dict = {'summer':{},'winter':{}}
transients_dict = {'summer':{},'winter':{}}


results_dic = {}
output_path = os.path.expanduser("~/Dropbox/niomtk_data/eco/downsampled")

classifiers_dict = {"SVM": svm.SVC(), "KNN": KNeighborsClassifier(n_neighbors=3),
                    "DT": tree.DecisionTreeClassifier(),"RF":RandomForestClassifier()}

metric_dict = {"Precision":precision_score, "Recall":recall_score,
               "MCC":matthews_corrcoef, "Accuracy":tp}

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
                for metric_name, metric_func in metric_dict.iteritems():
                    out[season][home][clf_name][metric_name] = metric_func(y_test_dict[season][home], pred)

    return out


def nilm_pair_remove_fridge(home, season, folder_path):
    from copy import deepcopy
    season_path = os.path.join(folder_path, season)
    home_path = season_path+"/"+str(home)+".csv"
    df = pd.read_csv(home_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    df = df.between_time("06:00", "22:00").copy()
    train_size = len(df)/2
    df = df.tail(train_size)
    from nilmtk.feature_detectors.steady_states import find_steady_states, find_steady_states_transients
    from nilmtk.disaggregate.hart_85 import Hart85
    h = Hart85()

    fridge_min = fridge_power[home]-30
    fridge_max = fridge_power[home]+30

    # Train on DF to get pairs.
    # 1. Between pairs put occupancy as 1
    # 2. If not a pair, then also put occupancy as 1
    ss, tr = find_steady_states(df[["power"]])
    pred = pd.Series(np.zeros(len(df)),name="occupancy", index=df.index)


    # Find unique days
    days = pd.DatetimeIndex(np.unique(df.index.date))
    for day in days:
        events_day = tr[day.strftime("%Y-%m-%d")].abs()
        # Find first non-fridge event
        event_df = events_day[(events_day['active transition']<=fridge_min)|(events_day['active transition']>=fridge_max)]

        if len(event_df)>0:
            first_event=event_df.index[0]
            last_event = event_df.index[-1]
            pred[day.strftime("%Y-%m-%d")][:first_event]=1
            pred[day.strftime("%Y-%m-%d")][last_event:]=1
    for ix, row in tr.iterrows():

        if not(fridge_min<=row['active transition']<=fridge_max):
            pred.ix[tr.index] = 1
    h.transients = deepcopy(tr)

    pair_df = h.pair(buffer_size=20,
              min_tolerance=100, percent_tolerance=0.035,
              large_transition=1000)
    pair_df_dict[season][home] = pair_df
    transients_dict[season][home] = h.transients
    for idx, row in pair_df.iterrows():
        start = row['T1 Time']
        end = row['T2 Time']
        if not(fridge_min<=row['T1 Active']<=fridge_max):
            time_delta = (end - start)/np.timedelta64(1, 'h')
            if time_delta<24:
                pred[start:end] = 1
            else:
                print "*"*80


    pred_resampled = pred.resample("15T", how="max")
    return pred_resampled
    gt_occupancy = df["occupancy"].resample("15T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred_resampled.index)
    return {"Accuracy":
            tp(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection]),
            "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection])
            }

def nilm_pair(home, season, folder_path, freq="15", flag=0):
    from copy import deepcopy
    season_path = os.path.join(folder_path, season)
    home_path = season_path+"/"+str(home)+".csv"
    df = pd.read_csv(home_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    df = df.between_time("06:00", "22:00")
    train_size = len(df)/2
    df = df.tail(train_size)
    gt_power[season][home] = df

    from nilmtk.feature_detectors.steady_states import find_steady_states, find_steady_states_transients
    from nilmtk.disaggregate.hart_85 import Hart85
    h = Hart85()

    # Train on DF to get pairs.
    # 1. Between pairs put occupancy as 1
    # 2. If not a pair, then also put occupancy as 1
    ss, tr = find_steady_states(df[["power"]])
    pred = pd.Series(np.zeros(len(df)),name="occupancy", index=df.index)

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
    gt_occupancy = df["occupancy"].resample(str(freq)+"T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred_resampled.index)
    if flag is 0:
        return {"Accuracy":
                tp(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection]),
                "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection])
                }
    else:
        return {}


def chen(home, season, folder_path):

    season_path = os.path.join(folder_path, season)
    home_path = season_path+"/"+str(home)+".csv"
    df = pd.read_csv(home_path, index_col=0)

    df.index = pd.to_datetime(df.index)

    pred_index = df.between_time("06:00", "22:00").resample("15T").index
    pred = pd.Series(np.zeros(len(pred_index)), name="occupancy",
                        index=pred_index)
    print pred
    print pred.index
    days = pd.DatetimeIndex(np.unique(df.index.date))
    for day in days:

        night_df = df[day.strftime("%Y-%m-%d")].between_time("01:00", "04:00")["power"]
        if len(night_df):
            avg_power_threshold = night_df.resample("15T").max()
            avg_range_threshold = (night_df.resample("15T", how="max")-night_df.resample("15T", how="min")).max()
            avg_sd_threshold = night_df.resample("15T", how="std").max()
        else:
            avg_power_threshold = df.median()
            avg_range_threshold = df.median()
            avg_sd_threshold = df.median()
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
    pred = pred_downsample.resample("15T").fillna(method='ffill').astype('int')
    return pred
    gt_occupancy = df["occupancy"].resample("15T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred.index)
    return {"Accuracy":
            tp(gt_occupancy.ix[index_intersection], pred.ix[index_intersection]),
            "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred.ix[index_intersection])
            }


def nilm_naive(home, season, folder_path):
    season_path = os.path.join(folder_path, season)
    home_path = season_path+"/"+str(home)+".csv"
    df = pd.read_csv(home_path, index_col=0)


    df.index = pd.to_datetime(df.index)

    df = df.between_time("06:00", "22:00")
    train_size = len(df)/2
    df = df.tail(train_size)
    from nilmtk.feature_detectors.steady_states import find_steady_states, find_steady_states_transients
    """
    X = df
    y = X.pop('occupancy')
    X_train_idx,X_test_idx,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
    X_train = X.ix[X_train_idx]
    X_test = X.ix[X_test_idx]
    """
    ss, tr = find_steady_states(df[["power"]])
    pred = pd.Series(np.zeros(len(df)),name="occupancy", index=df.index)

    pred.ix[tr.index] = 1
    pred_resampled = pred.resample("15T", how="max")
    return pred_resampled
    gt_occupancy = df["occupancy"].resample("15T", how="max").dropna()
    index_intersection = gt_occupancy.index.intersection(pred_resampled.index)
    return {"Accuracy":
            tp(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection]),
            "MCC":matthews_corrcoef(gt_occupancy.ix[index_intersection], pred_resampled.ix[index_intersection])
            }


def sensitivity_our_approach(folder_path):
    out = {}
    for season in ["summer", "winter"]:
        out[season] = {}
        season_path = os.path.join(folder_path, season)
        #for home in [1, 2, 3, 4, 5]:
        for home in  [4]:
            print home, season
            print "*"*80
            out[season][home] = {}
            home_path = season_path+"/"+str(home)+".csv"
            df = pd.read_csv(home_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.between_time("06:00", "22:00")
            for freq in [5, 10, 15, 20, 25]:
                out[season][home][freq] = nilm_pair(home, season,
                                                                   os.path.expanduser("~/Dropbox/niomtk_data/eco/1min/"),
                                                    freq)
    return out


def classify_train_test_same_home(folder_path):
    out = {}
    accuracy_dict = {}
    for season in ["summer", "winter"]:
        out[season] = {}

        accuracy_dict[season] = {}
        season_path = os.path.join(folder_path, season)
        for home in [1, 2, 3, 4, 5]:
        #for home in  [4]:
            print home, season
            print "*"*80
            out[season][home] = {}
            accuracy_dict[season][home] = {}
            home_path = season_path+"/"+str(home)+".csv"
            df = pd.read_csv(home_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            df = df.between_time("06:00", "22:00")
            X = df
            y = X.pop('occupancy')
            train_size = len(X)/2
            X_train,X_test,y_train,y_test = X.head(train_size), X.tail(train_size), y.head(train_size), y.tail(train_size)
            out[season][home]["gt"] = y_test.resample("15T", how="max")
            for clf_name, clf in classifiers_dict.iteritems():
                out[season][home][clf_name] = {}
                clf.fit(X_train, y_train)
                pred = pd.Series(clf.predict(X_test), name="occupancy", index=X_test.index)

                out[season][home][clf_name] = pred

            #out[season][home]["NILM naive"] = nilm_naive(home, season,
            #                                                       os.path.expanduser("~/Dropbox/niomtk_data/eco/1min/"))

            out[season][home]["Chen"] = chen(home, season,
                                                                   os.path.expanduser("~/Dropbox/niomtk_data/eco/1min/"))


            #out[season][home]["NILM pairing"] = nilm_pair(home, season,
            #                                                       os.path.expanduser("~/Dropbox/niomtk_data/eco/1min/"))

            out[season][home]["Our approach"] = nilm_pair_remove_fridge(home, season,
                                                                   os.path.expanduser("~/Dropbox/niomtk_data/eco/1min/"))

            #out[season][home]["Always occupied"] = pd.Series(np.ones(len(X_test)), name="occupancy", index=X_test.index)

            for pred_key in out[season][home].keys():
                if pred_key!="gt":
                    accuracy_dict[season][home][pred_key] = compute_metrics(out[season][home]["gt"],out[season][home][pred_key] )



    return out, accuracy_dict


def compute_metrics(gt, pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    gt = gt.copy().dropna()
    pred = pred.copy().dropna()
    index_intersection = gt.index.intersection(pred.index)
    gt_x = gt.ix[index_intersection]
    pred_x = pred.ix[index_intersection]
    tp = ((gt_x==1)&(pred_x==1)).sum()*1.0/((gt_x==1).sum())
    tn = ((gt_x==0)&(pred_x==0)).sum()*1.0/((gt_x==0).sum())
    fp = ((gt_x==0)&(pred_x==1)).sum()*1.0/((gt_x==0).sum())
    fn= ((gt_x==1)&(pred_x==0)).sum()*1.0/((gt_x==1).sum())
    accuracy = (gt_x==pred_x).sum()*1.0/len(gt_x)
    mcc = matthews_corrcoef(y_pred=pred_x, y_true=gt_x)
    return {"tp":tp, "tn":tn, "fp":fp, "fn":fn, "accuracy":accuracy,"MCC":mcc}



def plot_train_single_averaged(accuracy_dict=None):
    metric_plot="Accuracy"
    if accuracy_dict is None:
        out, accuracy_dict = classify_train_test_same_home(output_path)
    res = {}
    for season, season_dict in accuracy_dict.iteritems():
        res[season] = {}
        for metric in ['tp','tn','fp','fn','accuracy','MCC']:
            res[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                temp = pd.DataFrame(home_results).T.to_dict()
                if metric in temp:
                    res[season][metric][home] = temp[metric]
        # Plotting

    summer_dict = {}
    winter_dict = {}
    for metric in ['tp','tn','fp','fn','accuracy','MCC']:
        summer_dict[metric] = pd.DataFrame(res['summer'][metric]).T.mean()
        winter_dict[metric] = pd.DataFrame(res['winter'][metric]).T.mean()

        summer_df = pd.DataFrame(summer_dict).T
        summer_df = summer_df[['Chen', 'Our approach',  'KNN', 'SVM', 'RF']]

        winter_df = pd.DataFrame(winter_dict).T
        winter_df = winter_df[['Chen', 'Our approach',  'KNN', 'SVM', 'RF']]

    latexify(columns=2)

    ax = summer_df.plot(kind='bar', rot=0)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    format_axes(ax)
    plt.savefig(os.path.expanduser("~/git/NIOMTK/figures/eco_summer.pdf"), bbox_inches="tight")
    plt.clf()
    ax = winter_df.plot(kind='bar', rot=0)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    format_axes(ax)
    plt.savefig(os.path.expanduser("~/git/NIOMTK/figures/eco_winter.pdf"), bbox_inches="tight")

    plot_tp_fp_fn(summer_df.copy(), os.path.expanduser("~/git/NIOMTK/figures/eco_summer_scatter.pdf"))
    plot_tp_fp_fn(winter_df.copy(), os.path.expanduser("~/git/NIOMTK/figures/eco_winter_scatter.pdf"))


def plot_tp_fp_fn(results_df, plot_name):
    plt.clf()
    results_df.loc['tp+fp'] = results_df.loc['tp'] + results_df.loc['fp']
    y = results_df.ix['tp+fp']
    x = results_df.ix['fn']

    for i,(x_p, y_p) in enumerate(zip(x, y)):
        plt.plot(x_p, y_p, marker='o', linestyle='', label=x.index[i],ms=6)
    plt.legend(numpoints=1, loc='upper right')
    plt.xlabel("FN")
    plt.ylabel("TP+FP")
    format_axes(plt.gca())
    plt.savefig(plot_name,bbox_inches="tight")


def plot_train_single(metric_plot="Accuracy", out=None):
    if out is None:
        out = classify_train_test_same_home(output_path)
    res = {}
    for season, season_dict in out.iteritems():
        res[season] = {}
        for metric in ['tp','tn','fp','fn','accuracy','MCC']:
            res[season][metric] = {}
            for home, home_results in season_dict.iteritems():
                temp = pd.DataFrame(home_results).T.to_dict()
                if metric in temp:
                    res[season][metric][home] = temp[metric]
        # Plotting

    summer_df = pd.DataFrame(res['summer'][metric_plot]).T
    winter_df = pd.DataFrame(res['winter'][metric_plot]).T

    summer_df = summer_df.drop('gt',1)
    winter_df = winter_df.drop('gt',1)

    latexify(columns=2, fig_height=4)
    fig, ax = plt.subplots(nrows=2, sharex=True)

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




