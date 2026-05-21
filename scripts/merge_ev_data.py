import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from scipy.interpolate import interp1d
from tqdm import tqdm

# file paths
RAW_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "raw_data")
CHARGING_FILE = os.path.join(RAW_DATA_FOLDER, "Dataset1_charging_reports.csv")
PRICE_FILE = os.path.join(RAW_DATA_FOLDER, "Norway.csv")
OUTPUT_FILE = os.path.join(RAW_DATA_FOLDER, "EV_Charging_Data.csv")

# locations from data frame and their coordinates
location_coordinates = {
    "ASK": (59.83, 10.43),
    "BAR": (59.56, 10.30),
    "OSL_S": (59.54, 10.45),
    "OSL_T": (59.91, 10.83),
    "TRO": (63.44, 10.42),
    "TRO_R": (63.44, 10.42),
    "BER": (60.23, 5.19),
    "BOD": (67.16, 14.24),
    "KRO": (59.86, 10.78),
    "OSL_1": (59.54, 10.45),
    "OSL_2": (59.54, 10.45),
}


# source for weather data
API_URL = "https://archive-api.open-meteo.com/v1/era5"

def load_and_merge_data(
    charging_file=CHARGING_FILE,
    price_file=PRICE_FILE,
    output_file=OUTPUT_FILE,
    delimiter_charging=';',
    quotechar_charging='"'
):
    if os.path.exists(output_file):
        print(f"Loading existing merged data from {output_file}.")
        df = pd.read_csv(output_file, parse_dates=["plugin_time", "plugout_time"])
        df = df[df["location"].isin(location_coordinates.keys())]

    else:
        # load charging data
        df = pd.read_csv(charging_file, delimiter=delimiter_charging, quotechar=quotechar_charging)
        df.dropna(subset=["plugin_time", "plugout_time"], inplace=True)
        df["plugin_time"] = pd.to_datetime(df["plugin_time"])
        df["plugout_time"] = pd.to_datetime(df["plugout_time"])

        # time features
        df["plugin_minute"] = df["plugin_time"].dt.minute
        df["plugin_hour"] = df["plugin_time"].dt.hour
        df["plugin_day"] = df["plugin_time"].dt.day
        df["plugin_month"] = df["plugin_time"].dt.month
        df["plugin_weekday"] = df["plugin_time"].dt.weekday

        df["plugout_minute"] = df["plugout_time"].dt.minute
        df["plugout_hour"] = df["plugout_time"].dt.hour
        df["plugout_day"] = df["plugout_time"].dt.day
        df["plugout_month"] = df["plugout_time"].dt.month
        df["plugout_weekday"] = df["plugout_time"].dt.weekday

        df["location"] = df["location"].replace("BAR_2", "BAR")

        for col in ["connection_time", "energy_session"]:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(",", ".").astype(float)

        location_mapping = {
            "ASK": "sub-urban", "BAR": "sub-urban", "KRO": "rural",
            "OSL_S": "urban", "OSL_T": "urban", "OSL_1": "urban", "OSL_2": "urban",
            "TRO": "urban", "TRO_R": "urban", "BER": "urban", "BOD": "urban",
        }
 
        df["area_type"] = df["location"].map(location_mapping)

        # electricity prices
        prices_df = pd.read_csv(price_file)
        prices_df["Datetime (Local)"] = pd.to_datetime(prices_df["Datetime (Local)"])
        prices_df["Price (EUR/kWh)"] = prices_df["Price (EUR/MWhe)"] / 1000
        prices_df = prices_df.set_index("Datetime (Local)").sort_index()

        def get_hourly_block_price(timestamp: pd.Timestamp) -> float:
            return prices_df["Price (EUR/kWh)"].asof(timestamp)
        
        for index, row in df.iterrows():
            start_time = row["plugin_time"]
            start_hour = start_time.replace(minute=0, second=0, microsecond=0)
            price = get_hourly_block_price(start_hour)
            df.at[index, "electricity_price"] = price


    # add weather data
    df = merge_weather_data(df)

    # save
    df.to_csv(output_file, index=False)
    print(f"Saved merged data to {output_file}")
    return df


# weather fetching functions

def fetch_weather_data(lat, lon, start_time):
    start_date = start_time.date()
    end_date = start_time.date()  
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),  
        "hourly": "temperature_2m,relativehumidity_2m,shortwave_radiation,wind_speed_10m",  
    }

    try:
        response = requests.get(API_URL, params=params, timeout=10)
        if response.status_code == 429:
            print("Rate limit reached.")
        response.raise_for_status()
        data = response.json()

        if "hourly" in data:
            target_hour = start_time.hour
            return (
                data["hourly"]["temperature_2m"][target_hour],
                data["hourly"]["relativehumidity_2m"][target_hour],
                data["hourly"]["shortwave_radiation"][target_hour],
                data["hourly"]["wind_speed_10m"][target_hour]  
            )
    except Exception as e:
        print(f"Weather fetch error: {e}")
    return None, None, None, None 

def merge_weather_data(df):
    print("Fetching weather data from Open-Meteo ERA5.")

    if "temperature" not in df.columns:
        df["temperature"] = np.nan
    if "humidity" not in df.columns:
        df["humidity"] = np.nan
    if "solar_radiation" not in df.columns:
        df["solar_radiation"] = np.nan
    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    weather_cache = {}
    BATCH_SIZE = 10000
    counter = 0

    missing_rows = df[df["temperature"].isna()].copy()
    print(f"Found {len(missing_rows)} rows missing weather data.")

    for idx, row in tqdm(missing_rows.iterrows(), total=len(missing_rows), desc="Weather Fetch"):
        location = row["location"]
        start_time = row["plugin_time"]


        lat, lon = location_coordinates[location]
        cache_key = (lat, lon, start_time.date())

        if cache_key not in weather_cache:
            try:
                temp, hum, rad, wind = fetch_weather_data(lat, lon, start_time)
                if temp is None:
                    print(f"Rate limit reached. ")
                    break  
                weather_cache[cache_key] = (temp, hum, rad, wind) 
            except Exception as e:
                print(f"Weather fetch failed: {e}")
                break
        else:
            temp, hum, rad, wind = weather_cache[cache_key]

        df.at[idx, "temperature"] = temp
        df.at[idx, "humidity"] = hum
        df.at[idx, "solar_radiation"] = rad
        df.at[idx, "wind_speed"] = wind

        counter += 1
        if counter >= BATCH_SIZE:
            print(f"Limit reached: BATCH_SIZE={BATCH_SIZE}. Saving results.")
            break

    print("Weather data merged.")
    return df


if __name__ == "__main__":
    load_and_merge_data()
