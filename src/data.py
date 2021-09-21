import torch
import numpy as np
import random
import jax
import jax.numpy as jnp
import numbers
from data_read_parse import get_data
from data_types import Model_Params

def get_dataloaders(mp: Model_Params):
    class bus_dataset(torch.utils.data.Dataset):
        def __init__(self, trips):
            self.trips = trips

        def __len__(self):
            return len(self.trips)

        def __getitem__(self, idx):
            trip = self.trips[idx]
            r = random.randint(0, len(trip) - 1)
            zeroed = data_info.cat_names + data_info.non_cat_names
            trip.loc[trip.index < r, zeroed] = 0

            non_category = trip[data_info.non_cat_names].to_numpy().astype(np.float32)
            category = trip[data_info.cat_names].to_numpy().astype(np.int32)
            r = np.int32(r)
            label = trip[data_info.label_names].to_numpy().astype(np.float32)

            return (non_category, category, r), label
            # return non_category, label

    # https://krisztiankovacs.com/blog/deep_learning/jax/2021/05/03/Pytorch_Dataloaders_for_Jax.html
    class RandomSampler(torch.utils.data.Sampler):
        def __init__(self, data_source, rng_key):
            self.data_source = data_source
            self.rng_key = rng_key

        def __len__(self):
            return len(self.data_source)

        def __iter__(self):
            self.rng_key, current_rng = jax.random.split(self.rng_key)
            return iter(
                jax.random.permutation(current_rng, jnp.arange(len(self))).tolist()
            )

    def stack_collate(batch):
        """
        if it's first is an nd array, stack it and return it
        go through each thing,
                call this on each ?stack?

        how to we get these "stacks" out?
        we iterate trhough the current level, adding each index to it's own stack of things
        then we call the funciton on the stacks
        """
        first = batch[0]
        if isinstance(first, (np.ndarray, numbers.Number)):
            return np.stack(batch)
        else:
            number_subs = len(first)
            holder = tuple(list() for _ in range(number_subs))
            for example in batch:
                for i, input in enumerate(example):
                    holder[i].append(input)
            return tuple(stack_collate(x) for x in holder)

    class JaxDataLoader(torch.utils.data.DataLoader):
        def __init__(
            self,
            dataset,
            rng_key,
            batch_size=1,
            shuffle=False,
            collate_fn=stack_collate,
            **kwargs
        ):

            if shuffle:
                sampler = RandomSampler(dataset, rng_key)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)

            super().__init__(
                dataset, batch_size, sampler=sampler, collate_fn=collate_fn, **kwargs
            )

    worker_count = 1

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

    rng = jax.random.PRNGKey(0)

    train_loader = JaxDataLoader(
        train_data,
        rng,
        mp.batch_size,
        shuffle=True,
        collate_fn=stack_collate,
        num_workers=worker_count,
    )
    val_loader = JaxDataLoader(
        val_data,
        rng,
        mp.batch_size,
        shuffle=False,
        collate_fn=stack_collate,
        num_workers=worker_count,
    )
    test_loader = JaxDataLoader(
        test_data,
        rng,
        mp.batch_size,
        shuffle=False,
        collate_fn=stack_collate,
        num_workers=worker_count,
    )
    return train_loader, val_loader, test_loader, data_info
    # return dl