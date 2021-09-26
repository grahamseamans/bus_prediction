import os
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm
import config

def get_data():

    data_dir = os.path.join(os.getcwd(), "data")
    processed_dir = os.path.join(data_dir, "processed_data")
    file = f"trips_direction_{config.trip_direction}.npy"

    if config.recompute:

        pickle_path = os.path.join(data_dir, "mega_pickle")
        df = pd.read_pickle(pickle_path)
        df = df.head(1000000)

        df = df[df["direction"] == config.trip_direction]

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
        config.cat_names = [
            "direction",
            "door",
            "lift",
            "location_id",
            "schedule_status",
            "service_key",
            "train",
            "vehicle_number",
        ]
        config.time_names = [
            "arrival_timestamp",
            "stop_timestamp",
            "leave_timestamp",
        ]
        config.non_cat_names = [
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
        config.label_names = ["arrival_deviance"]

        df[config.time_names] = df[config.time_names].astype(int)
        df[config.time_names] = (
            df[config.time_names] - df[config.time_names].mean()
        ) / df[config.time_names].std()

        # just one timeststamp for now...
        config.time_names = ["stop_timestamp"]

        enc = OrdinalEncoder()
        df[config.cat_names] = enc.fit_transform(df[config.cat_names])
        df[config.cat_names] = df[config.cat_names].astype(np.int32)
        config.cardinality = [len(cat) for cat in enc.categories_]

        # config.cardinality = []
        # for category in config.cat_names:
        #     config.cardinality.append(df[category].max())

        df[config.non_cat_names] = df[config.non_cat_names].mask(
            np.abs(
                (df[config.non_cat_names] - df[config.non_cat_names].mean(axis=0))
                / df[config.non_cat_names].std(axis=0)
            )
            > 4,
            np.nan,
        )
        df[config.non_cat_names] = df[config.non_cat_names].interpolate(axis=0)
        df[config.non_cat_names] = df[config.non_cat_names].ffill(axis=0)
        df[config.non_cat_names] = df[config.non_cat_names].bfill(axis=0)

        config.non_cat_names.remove("arrival_deviance")

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
        config.trip_length = len(cannonical_stop_seq)

        stop_order_dict = {}
        for i, stop in enumerate(cannonical_stop_seq):
            stop_order_dict[stop] = i

        for i, trip in tqdm(
            enumerate(trips),
            total=len(trips),
            unit="trip",
            desc="making cannonical trips",
        ):
            trip = trip.drop_duplicates("location_id")

            seq = np.array([cannonical_stop_seq, list(range(config.trip_length))])
            seq = pd.DataFrame(seq.T, columns=["location_id", "idx"])

            trip = trip.merge(seq, how="outer", on="location_id")
            trip = trip.dropna(subset=["idx"])
            trip = trip.sort_values(by="idx")
            trip = trip.reset_index(drop=True)
            trip = trip.drop(columns=["idx"])
            trip = trip.fillna(0)  # essential! the join was (reasonably) making nans!
            trips[i] = trip

        # datas = [trips, config]

        # for file_name, file in zip(files, datas):
        config.save()
        file_path = os.path.join(processed_dir, file)
        pickle.dump(trips, open(file_path, "wb"))

    else:
        # arrays = []
        # for file_name in files:
        file_path = os.path.join(processed_dir, file)
        trips = pickle.load(open(file_path, "rb"))
        # trips, config = arrays

    return trips
