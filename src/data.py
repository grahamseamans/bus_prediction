import numpy as np
import random
import numbers
from prep_cleaned import get_data
import config
import torch


def get_dataloaders():
    class bus_dataset(torch.utils.data.Dataset):
        def __init__(self, trips):
            self.trips = trips

        def __len__(self):
            return len(self.trips)

        def __getitem__(self, idx):
            trip = self.trips[idx]
            r = random.randint(0, len(trip) - 1)
            zeroed = config.cat_names + config.non_cat_names
            trip.loc[trip.index < r, zeroed] = 0

            non_category = trip[config.non_cat_names].to_numpy().astype(np.float32)
            category = trip[config.cat_names].to_numpy().astype(np.int32)
            r = np.array(r, dtype=np.int32)
            label = trip[config.label_names].to_numpy().astype(np.float32)

            return [non_category, category, r], label

    # https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    def numpy_collate(batch):
        if isinstance(batch[0], (np.ndarray, numbers.Number)):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    class NumpyLoader(torch.utils.data.DataLoader):
        def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
        ):
            super(self.__class__, self).__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=numpy_collate,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
            )

    worker_count = 12

    trips = get_data()
    dataset = bus_dataset(trips)
    print(f"There are a total of {len(dataset)} trips")

    train = 0.6
    val = 0.2
    test = 0.2
    np.testing.assert_almost_equal(train + val + test, 1)

    datalen = len(dataset)

    train_slice = slice(0, int(train * datalen))
    val_slice = slice(int(train * datalen), int((val + train) * datalen))
    test_slice = slice(int((val + train) * datalen), datalen)

    train = trips[train_slice]
    val = trips[val_slice]
    test = trips[test_slice]

    train_loader = NumpyLoader(
        bus_dataset(train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=worker_count,
    )
    val_loader = NumpyLoader(
        bus_dataset(val), batch_size=config.batch_size, num_workers=worker_count
    )
    test_loader = NumpyLoader(
        bus_dataset(test), batch_size=config.batch_size, num_workers=worker_count
    )

    return train_loader, val_loader, test_loader
