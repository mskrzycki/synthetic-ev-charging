import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file, output_file):
    # read the raw dataset
    print(f"Reading raw data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # convert time columns to datetime
    if "plugin_time" in df.columns:
        df["plugin_time"] = pd.to_datetime(df["plugin_time"], errors="coerce")
    if "plugout_time" in df.columns:
        df["plugout_time"] = pd.to_datetime(df["plugout_time"], errors="coerce")
    
    # filter out outlier sessions
    if "energy_session" in df.columns:
        df = df[df["energy_session"] > 0.5]
        df = df[df["energy_session"] <= 150]
    else:
        print("Warning: 'energy_session' column not found.")
    
    # filter out sessions with short connection time (< 2 minutes)
    if "connection_time" in df.columns:
        df = df[df["connection_time"] >= 2]
    else:
        print("Warning: 'connection_time' column not found.")
    
    # calendar features based on plugin and plugout time
    if "plugin_time" in df.columns:
        df["plugin_minute"] = df["plugin_time"].dt.minute
        df["plugin_hour"] = df["plugin_time"].dt.hour
        df["plugin_day"] = df["plugin_time"].dt.day
        df["plugin_month"] = df["plugin_time"].dt.month
        df["plugin_weekday"] = df["plugin_time"].dt.weekday  # Monday=0, Sunday=6
    if "plugout_time" in df.columns:
        df["plugout_minute"] = df["plugout_time"].dt.minute
        df["plugout_hour"] = df["plugout_time"].dt.hour
        df["plugout_day"] = df["plugout_time"].dt.day
        df["plugout_month"] = df["plugout_time"].dt.month
        df["plugout_weekday"] = df["plugout_time"].dt.weekday  # Monday=0, Sunday=6

    #cols_to_drop = ['user_id', 'session_id', 'plugin_time', 'plugout_time']
    #df = df.drop(columns=cols_to_drop, errors='ignore') 


    winter_cond = (
        ((df["plugin_month"] == 12) & (df["plugin_day"] >= 22)) |   # 22 Dec – 31 Dec
        ( df["plugin_month"].isin([1, 2])                      ) |  # Jan–Feb
        ((df["plugin_month"] == 3 ) & (df["plugin_day"] <= 20))     # 1 Mar – 20 Mar
)

    spring_cond = (
        ((df["plugin_month"] == 3) & (df["plugin_day"] >= 21)) |    # 21 Mar – 31 Mar
        ( df["plugin_month"].isin([4, 5])                     ) |   # Apr–May
        ((df["plugin_month"] == 6) & (df["plugin_day"] <= 21))      # 1 Jun – 21 Jun
    )

    summer_cond = (
        ((df["plugin_month"] == 6) & (df["plugin_day"] >= 22)) |    # 22 Jun – 30 Jun
        ( df["plugin_month"].isin([7, 8])                     ) |   # Jul–Aug
        ((df["plugin_month"] == 9) & (df["plugin_day"] <= 22))      # 1 Sep – 22 Sep
    )

    autumn_cond = (
        ((df["plugin_month"] == 9)  & (df["plugin_day"] >= 23)) |   # 23 Sep – 30 Sep
        ( df["plugin_month"].isin([10, 11])                   ) |   # Oct–Nov
        ((df["plugin_month"] == 12) & (df["plugin_day"] <= 21))     # 1 Dec – 21 Dec
    )

    df["season"] = np.select([spring_cond, summer_cond, autumn_cond, winter_cond],[0,1, 2,3])

    def weekday_group(day: int) -> str:
        if day < 4:
            return "Mon-Th"
        if day == 4:
            return "Friday"
        if day == 5:
            return "Saturday"
        return "Sunday"

    df['weekday_group'] = df['plugin_weekday'].apply(weekday_group)

    location_groups = {
        'ASK':   0,
        'TRO_R':   0,
        'TRO': 1,
        'OSL_S': 1,
        'BAR':   2,
        'OSL_T': 3
    }

    df['location_group'] = df['location'].map(location_groups)

    df = df[df["location"] != "KRO"].copy()
    df = df[df["location"] != "OSL_2"].copy()
    df = df[df["location"] != "OSL_1"].copy()
    df = df[df["location"] != "BER"].copy()
    df = df[df["location"] != "BOD"].copy()
    # save the preprocessed data as CSV
    print(f"Saving processed data to: {output_file}")
    df.to_csv(output_file, index=False)
    
    return df


def main():
    input_path  = "raw_data/EV_Charging_Data.csv"
    output_path = "raw_data/EV_Charging_Data_processed.csv"

    preprocess_data(input_path, output_path)

if __name__ == "__main__":
    main()