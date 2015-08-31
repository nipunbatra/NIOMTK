import pandas as pd
import numpy as np
from itertools import chain
import os


def convert_occupancy_df(df, freq="15min"):
    start_date = df.values[0][0]
    end_date = df.values[-1][0]

    index = pd.DatetimeIndex(start=start_date, periods=len(df) * 86400, freq='1s')
    index = index.tz_localize("UTC").tz_convert("CET")

    out = []
    for i in range(len(df)):
        out.append(df.ix[i].values[1:])
    out_1d = list(chain.from_iterable(out))

    df_new = pd.Series(out_1d, index=index)
    # If occupied even for 1 second, then save occupied
    df_resampled = df_new.resample(freq, how="max")
    return df_resampled


def create_raw_power_occupancy(csv_folder, hdf, out_folder):
    subfolders = os.listdir(csv_folder)
    subfolders = [x for x in subfolders if "DS" not in x]
    for folder in subfolders:

        home_path = os.path.join(csv_folder, folder)
        home_num_str = folder[1]
        print "Home number " + home_num_str
        for season in ["summer", "winter"]:
            print season
            csv_name = folder[:3] + season + ".csv"
            csv_path = os.path.join(home_path, csv_name)
            df = pd.read_csv(csv_path)
            df_resampled = convert_occupancy_df(df, "1T")
            start_date = df.values[0][0]
            end_date = df.values[-1][0]
            store = pd.HDFStore(hdf)
            power_df = store['/building%s/elec/meter1' % home_num_str]
            power_df = pd.DataFrame({"power":power_df['power']['active'][start_date:end_date]})
            power_df["occupancy"] = df_resampled
            power_df = power_df.dropna()
            folder_path = os.path.join(out_folder, season)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            power_df.to_csv(folder_path + "/" + home_num_str + ".csv")


def create_15min_csv(csv_folder, hdf, out_folder):
    """

    :param csv_folder: path to the folder where occupancy CSVs are stored
    :return:
    """
    subfolders = os.listdir(csv_folder)
    subfolders = [x for x in subfolders if "DS" not in x]
    for folder in subfolders:

        home_path = os.path.join(csv_folder, folder)
        home_num_str = folder[1]
        print "Home number " + home_num_str
        for season in ["summer", "winter"]:
            print season
            csv_name = folder[:3] + season + ".csv"
            csv_path = os.path.join(home_path, csv_name)
            df = pd.read_csv(csv_path)
            df_resampled = convert_occupancy_df(df)
            start_date = df.values[0][0]
            end_date = df.values[-1][0]
            store = pd.HDFStore(hdf)
            power_df = store['/building%s/elec/meter1' % home_num_str]
            power_df = power_df['power']['active'][start_date:end_date]
            out = {}
            for feature in ["min", "max", "mean", "std"]:
                out[feature] = power_df.resample("15T", how=feature)
            out["range"] = out["max"] - out["min"]
            out_df = pd.DataFrame(out)
            out_df["occupancy"] = df_resampled
            out_df = out_df.dropna()
            folder_path = os.path.join(out_folder, season)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            out_df.to_csv(folder_path + "/" + home_num_str + ".csv")


csv_path = os.path.expanduser("~/Dropbox/niomtk_data/eco/occupancy_raw/")
hdf = os.path.expanduser("~/Downloads/eco.h5")
output_path = os.path.expanduser("~/Dropbox/niomtk_data/eco/downsampled")
output_path_raw = os.path.expanduser("~/Dropbox/niomtk_data/eco/1min")

#create_15min_csv(csv_path, hdf, output_path)
create_raw_power_occupancy(csv_path, hdf, output_path_raw)
