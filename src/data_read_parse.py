import os
from data_types import Data_Info
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import OrdinalEncoder

data_dir = os.path.join(os.getcwd(), "data")


def get_data(recompute, direction):
    processed_dir = os.path.join(data_dir, "processed_data")
    trips = None
    files = [f"{f}_direction_{direction}.npy" for f in ["trips", "cardinality"]]

    data_info = Data_Info()

    cardinality = []

    if recompute:

        pickle_path = os.path.join(data_dir, "mega_pickle")
        df = pd.read_pickle(pickle_path)
        # df = df.head(1000)

        df = df[df["direction"] == direction]

        df = df.sort_values(["service_date", "train", "trip_number", "stop_time"])
        df = df.reset_index(drop=True)

        df = df.drop(
            columns=[
                "route_number",
                "arrive_time",
                "leave_time",
                "stop_time",
            ]
        )

        dtypes = df.dtypes
        print(dtypes)

        """
        data_source                               float32
        direction                                category
        door                                     category
        dwell                                       int64
        estimated_load                              int64
        lift                                     category
        location_id                              category
        maximum_speed                               int64
        offs                                        int64
        ons                                         int64
        pattern_distance                          float32
        schedule_status                          category
        service_date                       datetime64[ns]
        service_key                              category
        train                                    category
        train_mileage                             float32
        trip_number                                 int64
        vehicle_number                           category
        x_coordinate                              float32
        y_coordinate                              float32
        arrival_timestamp                  datetime64[ns]
        leave_timestamp                    datetime64[ns]
        stop_timestamp                     datetime64[ns]
        arrival_deviance                            int64
        arrive_deviance_departure_delta             int64
        """
        data_info.cat_names = [
            "direction",
            "door",
            "lift",
            "location_id",
            "schedule_status",
            "service_key",
            "train",
            "vehicle_number",
        ]
        data_info.time_names = [
            "arrival_timestamp",
            "stop_timestamp",
            "leave_timestamp",
        ]
        data_info.non_cat_names = [
            "dwell",
            "estimated_load",
            "maximum_speed",
            "offs",
            "ons",
            "pattern_distance",
            "train_mileage",
            "arrival_deviance",
            "arrive_deviance_departure_delta",
        ]
        data_info.label_names = ["arrival_deviance"]

        df[data_info.time_names] = df[data_info.time_names].astype(int)
        df[data_info.time_names] = (
            df[data_info.time_names] - df[data_info.time_names].mean()
        ) / df[data_info.time_names].std()

        # just one timeststamp for now...
        data_info.time_names = ["stop_timestamp"]

        enc = OrdinalEncoder()
        df[data_info.cat_names] = enc.fit_transform(df[data_info.cat_names])
        df[data_info.cat_names] = df[data_info.cat_names].astype(np.int32)
        data_info.cardinality = [len(cat) for cat in enc.categories_]

        # data_info.cardinality = []
        # for category in data_info.cat_names:
        #     data_info.cardinality.append(df[category].max())

        df[data_info.non_cat_names] = df[data_info.non_cat_names].mask(
            np.abs(
                (df[data_info.non_cat_names] - df[data_info.non_cat_names].mean(axis=0))
                / df[data_info.non_cat_names].std(axis=0)
            )
            > 4,
            np.nan,
        )
        df[data_info.non_cat_names] = df[data_info.non_cat_names].interpolate(axis=0)
        df[data_info.non_cat_names] = df[data_info.non_cat_names].ffill(axis=0)
        df[data_info.non_cat_names] = df[data_info.non_cat_names].bfill(axis=0)

        data_info.non_cat_names.remove("arrival_deviance")

        trips = df.groupby(["service_date", "train", "trip_number"])
        trips = [trip for _, trip in trips]

        df = df.drop(
            columns=[
                "service_date",
            ]
        )
        random.shuffle(trips)
        # WE CAN DO THIS BECAUSE WE'RE ONLY TRYING TO PREDICT THE NEXT TRIP

        hist = {}
        for trip in trips:
            trip = trip.drop_duplicates("location_id")
            s = tuple(trip["location_id"].to_list())
            if s in hist:
                hist[s] = hist[s] + 1
            else:
                hist[s] = 1

        sorted_stops = sorted(hist.items(), key=lambda x: x[1], reverse=True)
        cannonical_stop_seq = list(sorted_stops[0][0])[
            1:-1
        ]  # LAST STOP WAS REGULARLY ~13 MINUTES EARLY...
        data_info.trip_length = len(cannonical_stop_seq)

        stop_order_dict = {}
        for i, stop in enumerate(cannonical_stop_seq):
            stop_order_dict[stop] = i

        for i, trip in enumerate(trips):
            trip = trip.drop_duplicates("location_id")

            seq = np.array([cannonical_stop_seq, list(range(data_info.trip_length))])
            seq = pd.DataFrame(seq.T, columns=["location_id", "idx"])

            trip = trip.merge(seq, how="outer", on="location_id")
            trip = trip.dropna(subset=["idx"])
            trip = trip.sort_values(by="idx")
            trip = trip.reset_index(drop=True)
            trip = trip.drop(columns=["idx"])
            trips[i] = trip

        datas = [trips, data_info]

        for file_name, file in zip(files, datas):
            file_path = os.path.join(processed_dir, file_name)
            pickle.dump(file, open(file_path, "wb"))

    else:
        arrays = []
        for file_name in files:
            file_path = os.path.join(processed_dir, file_name)
            arrays.append(pickle.load(open(file_path, "rb")))
        trips, data_info = arrays

    return trips, data_info
