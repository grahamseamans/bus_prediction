import os
import random
import torch
import pytorch_lightning
from data_types import Model_Params
from matplotlib import pyplot as plt
from model import get_model
from data_read_parse import get_data

data_dir = os.path.join(os.getcwd(), "data")

mp = Model_Params()


class bus_dataset(torch.utils.data.Dataset):
    def __init__(self, trips):
        self.trips = trips

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        trip = self.trips[idx]
        r_int = random.randint(0, len(trip) - 1)
        zeroed = data_info.cat_names + data_info.non_cat_names
        trip.loc[trip.index < r_int, zeroed] = 0

        r = torch.tensor(r_int).int()
        non_category = torch.tensor(trip[data_info.non_cat_names].values).float()
        category = torch.tensor(trip[data_info.cat_names].values).int()
        label = torch.tensor(trip[data_info.label_names].values).float()

        label = label.squeeze()

        return (non_category, category, r), label


worker_count = 12

trips, data_info = get_data(recompute=True, direction=1)
dataset = bus_dataset(trips)

train = 0.8
val = 0.17
test = 0.03
assert round(train + val + test) == 1

datalen = len(dataset)

train = int(train * datalen)
val = int(val * datalen)
test = datalen - (train + val)

train_data, test_data = torch.utils.data.random_split(dataset, [train + val, test])
train_data, val_data = torch.utils.data.random_split(train_data, [train, val])

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=mp.batch_size,
    shuffle=True,
    num_workers=worker_count,
)
val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=mp.batch_size,
    shuffle=False,
    num_workers=worker_count,
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=mp.batch_size,
    shuffle=False,
    num_workers=worker_count,
)

model = get_model(mp, data_info, dataset)

early_stop_callback = pytorch_lightning.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=20, verbose=False, mode="min"
)

checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
    monitor="val_loss", mode="min"
)

trainer = pytorch_lightning.Trainer(
    gpus=1,
    callbacks=[early_stop_callback, checkpoint_callback],
    progress_bar_refresh_rate=2,
)

trainer.fit(model, train_loader, val_loader)

checkpoint_callback.best_model_path
plot_loader = val_loader
preds = trainer.predict(model, plot_loader)
plot_dir = os.path.join(data_dir, "plots")
plot_per_batch = 1
for i, (pred_batch, loader_batch) in enumerate(zip(preds, plot_loader)):
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
            plt.rcParams["figure.figsize"] = [20, 4]
            plt.scatter(x, pred, label="pred")
            plt.scatter(x, label, label="y")
            plt.scatter(list(range(num_samples)), samp_x, label="samp")
            plt.legend()
            plt.savefig(os.path.join(plot_dir, f"plot_{i}.png"))
            plt.close()
