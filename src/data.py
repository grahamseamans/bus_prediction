import numpy as np
import random

from tensorflow.python.keras.backend import batch_dot
from data_read_parse import get_data
from data_types import Data_Info, Model_Params
import tensorflow as tf


class bus_dataset(tf.keras.utils.Sequence):
    def __init__(self, trips: list, data_info: Data_Info, batch_size: int,
                 shuffle: bool):
        assert len(trips) > 0
        self.trips = trips
        self.data_info = data_info
        self.datalen = len(self.trips)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

        X, y = self.trip_to_np_arrays(self.trips[0])
        self.X_shapes = [x.shape for x in X]
        self.y_shapes = [y.shape for y in y]

    def __len__(self):
        return self.datalen // self.batch_size

    def trip_to_np_arrays(self, trip):
        r = random.randint(0, len(trip) - 1)
        zeroed = self.data_info.cat_names + self.data_info.non_cat_names
        trip.loc[trip.index > r, zeroed] = 0

        non_category = trip[self.data_info.non_cat_names].to_numpy().astype(
            np.float32)
        category = trip[self.data_info.cat_names].to_numpy().astype(np.int32)
        r = np.array(r, dtype=np.int32)
        label = trip[self.data_info.label_names].to_numpy().astype(np.float32)
        return [non_category, category, r], [label]

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) *
                               self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def __data_generation(self, batch_idxs):
        X = [np.empty((self.batch_size, *shape)) for shape in self.X_shapes]
        y = [np.empty((self.batch_size, *shape)) for shape in self.y_shapes]

        for i, trip_idx in enumerate(batch_idxs):
            trip_x, trip_y = self.trip_to_np_arrays(self.trips[trip_idx])
            for j in range(len(self.X_shapes)):
                X[j][i] = trip_x[j]
            for j in range(len(self.y_shapes)):
                y[j][i] = trip_y[j]

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def get_dataloaders(mp: Model_Params):
    trips, data_info = get_data(recompute=False, direction=1)

    train = 0.6
    val = 0.2
    test = 0.2
    np.testing.assert_almost_equal(train + val + test, 1)

    datalen = len(trips)
    train_slice = slice(0, int(train * datalen))
    val_slice = slice(int(train * datalen), int((train + val) * datalen))
    test_slice = slice(int((train + val) * datalen), datalen)

    train = trips[train_slice]
    val = trips[val_slice]
    test = trips[test_slice]

    print(type(trips[0]))
    print(type(train[0]))

    print(len(trips))
    print(len(train))
    print(len(val))
    print(len(test))

    train = bus_dataset(train,
                        data_info,
                        batch_size=mp.batch_size,
                        shuffle=True)
    val = bus_dataset(val, data_info, batch_size=mp.batch_size, shuffle=False)
    test = bus_dataset(test,
                       data_info,
                       batch_size=mp.batch_size,
                       shuffle=False)

    return train, val, test, data_info
