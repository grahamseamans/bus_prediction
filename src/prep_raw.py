from IPython import get_ipython
import os
import pandas as pd
import numpy as np
import zipfile
import shutil
import dask
import dask.dataframe as dd
import datetime
from functools import lru_cache
import datetime


data_dir = os.path.join(os.getcwd(), "data")
stop_events = os.path.join(data_dir, "stop_event")

for file in os.listdir(stop_events):
    file = os.path.join(stop_events, file)
    if zipfile.is_zipfile(file):
        print(f"extracting: {file}")
        with zipfile.ZipFile(file) as item:
            item.extractall(stop_events)

stop_event_files = os.listdir(stop_events)

for item in stop_event_files:
    if ".zip" in item or "census" in item:
        os.remove(os.path.join(stop_events, item))


def lower_case_sort_columns(df):
    df.columns = df.columns.str.lower()

    df = df[
        [
            "arrive_time",
            "data_source",
            "direction",
            "door",
            "dwell",
            "estimated_load",
            "leave_time",
            "lift",
            "location_id",
            "maximum_speed",
            "offs",
            "ons",
            "pattern_distance",
            "route_number",
            "schedule_status",
            "service_date",
            "service_key",
            "stop_time",
            "train",
            "train_mileage",
            "trip_number",
            "vehicle_number",
            "x_coordinate",
            "y_coordinate",
        ]
    ]
    return df


def parse_stop_event(df):

    month_dict = {
        "JAN": "01",
        "FEB": "02",
        "MAR": "03",
        "APR": "04",
        "MAY": "05",
        "JUN": "06",
        "JUL": "07",
        "AUG": "08",
        "SEP": "09",
        "OCT": "10",
        "NOV": "11",
        "DEC": "12",
    }

    @lru_cache()
    def parse_date(date_str):
        date = date_str.split(":")[0]
        day = date[:2]
        month = month_dict[date[2:5]]
        year = date[5:]
        return year + "/" + month + "/" + day

    df["service_date"] = df["service_date"].apply(parse_date)
    df["service_date"] = pd.to_datetime(df["service_date"], format="%Y/%m/%d")

    df["arrival_timestamp"] = df["service_date"] + pd.to_timedelta(
        df["arrive_time"], unit="s"
    )
    df["leave_timestamp"] = df["service_date"] + pd.to_timedelta(
        df["leave_time"], unit="s"
    )
    df["stop_timestamp"] = df["service_date"] + pd.to_timedelta(
        df["stop_time"], unit="s"
    )

    # df.drop(columns=["service_date", "arrive_time", "leave_time", "stop_time"])

    df["arrival_deviance"] = df["stop_time"] - df["arrive_time"]
    df["arrive_deviance_departure_delta"] = (
        df["arrival_deviance"] + df["leave_time"] - df["arrive_time"]
    )

    return df


stop_event_file_names = [
    "2 trimet_stop_event - Sep 2020 to Mar 2021.csv",
    "2 trimet_stop_event - Aug 31 and Sep 1 2020.csv",
    "2 trimet_stop_event - Fall 2019.csv",
    "2 trimet_stop_event - Mar to Aug 2020.csv",
    "2 trimet_stop_event - Spring 2019.csv",
    "2 trimet_stop_event - Summer 2019.csv",
    "2 trimet_stop_event - Winter 2018.csv",
    "2 trimet_stop_event - Winter 2019-20.csv",
    "trimet_stop_event - Fall 2018 v2.csv",
]


stop_event_dfs = []
for file_name in stop_event_file_names:
    print(f"working on {file_name}")
    file_name = os.path.join(stop_events, file_name)
    df = dd.read_csv(
        file_name,
        dtype={
            "LOCATION_DISTANCE": "float32",
            "PATTERN_DISTANCE": "float32",
            "TRAIN_MILEAGE": "float32",
            "X_COORDINATE": "float32",
            "Y_COORDINATE": "float32",
            "DATA_SOURCE": "float32",
            "LOCATION_ID": "float32",
            "SCHEDULE_STATUS": "float32",
        },
    )
    df = df[df["ROUTE_NUMBER"] == 9].compute()  # can I do this here?
    df = lower_case_sort_columns(df)
    df = parse_stop_event(df)
    stop_event_dfs.append(df)

df = pd.concat(stop_event_dfs, axis=0, ignore_index=True)

categories = [
    "vehicle_number",
    "train",
    "route_number",
    "direction",
    "service_key",
    "location_id",
    "door",
    "lift",
    "schedule_status"
]

df[categories] = df[categories].astype("category")

df = df.sort_values(["service_date", "train", "trip_number", "stop_time"])
df = df.reset_index(drop=True)

pickle_dir = os.path.join(data_dir, "mega_pickle")
df.to_pickle(pickle_dir)

print("finished")


# save_file = os.path.join(data_dir, "mega_stop_event_1.hdf")
# df = pd.read_hdf(save_file, "/df")

# pickle_dir = os.path.join(data_dir, "mega_pickle")
# df.to_pickle(pickle_dir)
