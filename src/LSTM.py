import numpy as np
import pandas as pd
import os
import sys
import pickle
from numba import njit
from functools import lru_cache
from IPython.display import Audio, display
from sklearn.preprocessing import OrdinalEncoder
import random
import torch
import pytorch_lightning

from matplotlib import pyplot as plt


data_dir = os.path.join(os.getcwd(), "data")

def get_data(recompute, direction):
    # data_dir = os.path.join(os.getcwd(), "data")
    lstm_dir = os.path.join(data_dir, "trips_for_lstm")

    trips, dates, labels = None, None, None
    files = [
        f"{f}_direction_{direction}.npy"
        for f in ["non_categories", "categories", "labels", "cardinality"]
    ]

    cardinality = []

    if recompute:

        pickle_path = os.path.join(data_dir, "mega_pickle")
        df = pd.read_pickle(pickle_path)
        #     df = df.head(100000)

        df = df.drop(
            columns=["route_number", "time_cat_stop_time", "time_cat_leave_time"]
        )

        df = df.sort_values(["service_date", "train", "trip_number", "stop_time"])
        df = df.reset_index(drop=True)

        df = df[df["direction"] == direction]

        dtypes = df.dtypes
        category_names = dtypes[dtypes == "category"]
        category_names = category_names.index.to_list()
        non_category_names = dtypes[dtypes != "category"]
        non_category_names = non_category_names.index.to_list()
        non_category_names.remove("service_date")
        # non_category_names.remove("arrival_deviance")
        label_names = ["arrival_deviance"]

        enc = OrdinalEncoder()
        df[category_names] = enc.fit_transform(df[category_names])
        df[category_names] = df[category_names].astype(np.int32)

        df[non_category_names] = (
            df[non_category_names] - df[non_category_names].mean()
        ) / df[non_category_names].std()

        non_category_names.remove("arrival_deviance")

        for category in category_names:
            cardinality.append(len(df[category].unique()))

        trips = df.groupby(["service_date", "train", "trip_number"])
        trips = [trip for _, trip in trips]

        random.shuffle(
            trips
        )  # WE CAN DO THIS BECAUSE WE'RE ONLY TRYING TO PREDICT THE NEXT TRIP

        hist = {}
        for trip in trips:
            trip = trip.drop_duplicates("location_id")
            s = tuple(trip["location_id"].to_list())
            if s in hist:
                hist[s] = hist[s] + 1
            else:
                hist[s] = 1

        sorted_stops = sorted(hist.items(), key=lambda x: x[1], reverse=True)
        most_common_stop_sequence = list(sorted_stops[0][0])[
            1:-1
        ]  # LAST STOP WAS REGULARLY ~13 MINUTES EARLY...
        cannonical_length = len(most_common_stop_sequence)

        stop_order_dict = {}
        for i, stop in enumerate(most_common_stop_sequence):
            stop_order_dict[stop] = i

        x1 = []
        x2 = []
        y = []
        for trip in trips:
            trip = trip.drop_duplicates("location_id")

            labels = trip[label_names]
            categories = trip[category_names]
            non_categories = trip[non_category_names]

            cats = np.array(categories)
            non_cats = np.array(non_categories)
            l = np.array(labels)

            c_cats = np.zeros((cannonical_length, cats.shape[1]))
            c_non_cats = np.zeros((cannonical_length, non_cats.shape[1]))
            c_l = np.zeros(cannonical_length)

            loc_id_index = category_names.index("location_id")

            this_trips_stops = cats[:, loc_id_index].tolist()

            for i, loc_id in enumerate(most_common_stop_sequence):
                if loc_id in this_trips_stops:
                    idx = this_trips_stops.index(loc_id)
                    c_cats[i] = cats[idx]
                    c_non_cats[i] = non_cats[idx]
                    c_l[i] = l[idx]

            x1.append(c_non_cats)
            x2.append(c_cats)
            y.append(c_l)

        non_categories = np.stack(x1)
        categories = np.stack(x2)
        labels = np.stack(y)
        cardinality = np.array(cardinality)

        arrays = [non_categories, categories, labels, cardinality]

        for file_name, array in zip(files, arrays):
            file_path = os.path.join(lstm_dir, file_name)
            np.save(file_path, array)

    else:
        arrays = []
        for file_name in files:
            file_path = os.path.join(lstm_dir, file_name)
            arrays.append(np.load(file_path))
        non_categories, categories, labels, cardinality = arrays

    return non_categories, categories, labels, cardinality


class bus_dataset(torch.utils.data.Dataset):
    def __init__(self, recompute, direction, batch_size):

        self.non_categories, self.categories, self.labels, self.cardinality = get_data(
            recompute, direction
        )

        self.batch_size = batch_size

        self.datalen = self.non_categories.shape[0]
        self.num_stops = self.non_categories.shape[1]
        self.non_category_width = self.non_categories.shape[2]
        self.category_width = self.categories.shape[2]

        self.non_category_shape = (self.num_stops, self.non_category_width)
        self.category_shape = (self.num_stops, self.category_width)
        self.label_shape = (self.num_stops,)

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        r_int = random.randint(0, self.num_stops)
        r = torch.tensor(r_int).int()
        non_category = torch.tensor(self.non_categories[idx]).float()
        category = torch.tensor(self.categories[idx]).int()
        label = torch.tensor([self.labels[idx]]).float()

        non_category[r_int:] = 0
        category[r_int:] = 0

        label = torch.transpose(label, 0, 1)

        return (non_category, category, r), label


batch_size = 32
worker_count = 12

dataset = bus_dataset(recompute=False, direction=1, batch_size=batch_size)

train = 0.6
val = 0.3
test = 0.1
assert round(train + val + test) == 1

datalen = len(dataset)

train = int(train * datalen)
val = int(val * datalen)
test = datalen - (train + val)

print(train, val, test)


train_data, test_data = torch.utils.data.random_split(dataset, [train + val, test])
train_data, val_data = torch.utils.data.random_split(train_data, [train, val])

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=worker_count,
    # pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=worker_count,
    # pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=worker_count,
    # pin_memory=True,
)

print("made dataloaders")


print(dataset.non_category_width)
print(dataset.non_category_shape)

drop_rate = 0.2


def normalize_on_axis(x, normalizer, axis):
    x = x.permute(0, axis, 1)
    x = normalizer(x)
    x = x.permute(0, axis, 1)
    return x


class LSTM(pytorch_lightning.core.lightning.LightningModule):
    def __init__(self, data, learning_rate):
        super().__init__()

        self.data = data
        self.learning_rate = learning_rate
        self.lr = learning_rate

        self.hidden_width = 64
        self.embedding_scale = 4

        self.embedding_outs = [
            card // self.embedding_scale + 2 for card in data.cardinality
        ]
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(card, out_size)
                for card, out_size in zip(data.cardinality, self.embedding_outs)
            ]
        )
        self.total_cat_out = sum(self.embedding_outs)
        self.cat_and_non_cat_width = self.total_cat_out + self.data.non_category_width

        self.norm = torch.nn.BatchNorm1d(self.cat_and_non_cat_width)

        # self.lstm_1 = torch.nn.LSTM(
        #     self.cat_and_non_cat_width, self.lstm_1_width, batch_first=True, num_layers=4
        # )

        self.gru = torch.nn.GRU(
            self.cat_and_non_cat_width,
            self.hidden_width,
            batch_first=True,
            num_layers=4,
        )

        self.conv_1 = torch.nn.Conv1d(self.cat_and_non_cat_width, 16, 16, padding='valid')
        self.conv_2 = torch.nn.Conv1d(16, 1, 16, padding='valid')

        # self.lstm_to_outs = torch.nn.Linear(self.hidden_width, 1)

        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        non_category, category, r = x

        x = category
        x = torch.unbind(x, dim=2)
        outs = [embedding(cat) for cat, embedding in zip(x, self.embeddings)]

        x = torch.cat([*outs, non_category], dim=2)
        print(x.shape)
        x = x.permute(0,2,1)
        print(x.shape)
        # x = normalize_on_axis(x, self.norm, 2)
        x = self.norm(x)
        print(x.shape)

        # x, state = self.lstm_1(x)
        # x, state = self.gru(x)
        print(x.shape)
        x = self.conv_1(x)
        x = self.conv_2(x)
        print('post conv', x.shape)
        assert 2 == 3
        # x = self.lstm_to_outs(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))


model = LSTM(data=dataset, learning_rate=0.001)

early_stop_callback = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
)

checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
    monitor="val_loss", mode="min"
)

trainer = pytorch_lightning.Trainer(
    gpus=1,
    callbacks=[early_stop_callback, checkpoint_callback],
    progress_bar_refresh_rate=2,
    auto_lr_find=True,
)

# trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)

trainer.fit(model, train_loader, val_loader)

checkpoint_callback.best_model_path

preds = trainer.predict(model, test_loader)
plot_dir = os.path.join(data_dir, 'plots')
plot_per_batch = 1
for i, (pred_batch, loader_batch) in enumerate(zip(preds, test_loader)):
    if i % 5 == 0:
        (non_category_batch, category_batch, r_batch), label_batch = loader_batch
        for r, label, pred in zip(
            r_batch[:plot_per_batch],
            label_batch[:plot_per_batch],
            pred_batch[:plot_per_batch],
        ):
            pred = torch.squeeze(pred).cpu().numpy()
            label = torch.squeeze(label).cpu().numpy()
            r = int(r.item())

            assert len(label) == len(pred)
            x = list(range(len(pred)))

            num_samples = r
            samp_x = [0] * num_samples
            # print(pred)
            # print(label)
            # plt.rcParams["figure.figsize"] = [20, 4]
            plt.scatter(x, pred, label="pred")
            plt.scatter(x, label, label="y")
            plt.scatter(list(range(num_samples)), samp_x, label="samp")
            plt.legend()
            plt.savefig(os.path.join(plot_dir, f"plot_{i}.png"))
            plt.close()
