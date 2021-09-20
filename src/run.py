import os
import random
import torch
from data_types import Model_Params
from matplotlib import pyplot as plt
from model import get_model
from data_read_parse import get_data
from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Generator, Mapping, Tuple

Batch = Tuple
mp = Model_Params()


def get_dataloaders():
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

            non_category = trip[data_info.non_cat_names].to_numpy()
            category = trip[data_info.cat_names].to_numpy()
            r = r
            label = trip[data_info.label_names].to_numpy()

            # return (non_category, category, r), label
            return non_category, label

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
        # we're just stacking to collate
        # I think that we need to be doing something different to have mutiple model inputs
        # just add another forloop to the forloop..., should work...
        return [np.stack(x) for x in zip(*batch)]

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
    dl = JaxDataLoader(
        dataset, rng, 8, shuffle=True, collate_fn=stack_collate, num_workers=4
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=mp.batch_size,
    #     shuffle=True,
    #     num_workers=worker_count,
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_data,
    #     batch_size=mp.batch_size,
    #     shuffle=False,
    #     num_workers=worker_count,
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_data,
    #     batch_size=mp.batch_size,
    #     shuffle=False,
    #     num_workers=worker_count,
    # )
    # return train_loader, val_loader, test_loader
    return dl


def net_fn(batch: Batch) -> jnp.ndarray:
    print("batch shape", len(batch))
    # (non_category, category, r) = batch
    non_category = batch
    # https://dm-haiku.readthedocs.io/en/latest/api.html#causal
    mlp = hk.Sequential(
        [
            hk.Flatten(),
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(100),
            jax.nn.relu,
            hk.Linear(80),
        ]
    )
    return mlp(non_category)


# https://github.com/deepmind/optax/blob/master/examples/quick_start.ipynb


def main(_):

    # train_loader, val_loader, test_loader = get_dataloaders()
    dl = get_dataloaders()

    net = hk.without_apply_rng(hk.transform(net_fn))
    xs, ys = next(iter(dl))
    params = net.init(jax.random.PRNGKey(42), xs)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    def compute_loss(params, x, y):
        y_pred = net.apply(params, x)
        print(y_pred.shape)
        y = jnp.squeeze(y)
        print(y.shape)
        print("BBBBBBBBBBBBBBB\nBBBBBBBBBBBBBB\nBBBBBBBBBBBBBB\nBBBBBBBBBBBBB\n")
        loss = jnp.mean(optax.l2_loss(y_pred, y))
        print(loss)
        return loss

    for xs, ys in dl:
        grads = jax.grad(compute_loss)(params, xs, ys)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)


# def types_dataloader(batch):
#     # i think i could do this with a jax tree and map
#     print("BBBBBBBBBBBBBBB\nBBBBBBBBBBBBBB\nBBBBBBBBBBBBBB\nBBBBBBBBBBBBB\n")
#     (non_category, category, r), label = batch
#     non_category = torch_to_jnp(non_category, jnp.float32)
#     category = torch_to_jnp(category, jnp.int32)
#     print(type(r))
#     print(r.shape)
#     r = torch_to_jnp(r, jnp.int32)
#     print(r.shape)
#     label = torch_to_jnp(label, jnp.float32)
#     return (non_category, category, r), label


# def torch_to_jnp(tensor, type):
#     return tensor.numpy().astype(type)


if __name__ == "__main__":
    app.run(main)

"""
I can just run preds on test_data that I get out of randomsplit
that way I'll just get a massive np array that I can do whatever I want with?
I'd need to figure out how to turn a DataSet into two numpy arrays...

https://stackoverflow.com/questions/54897646/pytorch-datasets-converting-entire-dataset-to-numpy
the second index is the labels
"""


# plot_loader = val_loader
# # preds = trainer.predict(model, plot_loader)
# plot_dir = os.path.join(data_dir, "plots")
# plot_per_batch = 1
# for i, (pred_batch, loader_batch) in enumerate(zip(preds, plot_loader)):
#     if i % 5 == 0:
#         (non_category_batch, category_batch, r_batch), label_batch = loader_batch
#         for r, label, pred in zip(
#             r_batch[:plot_per_batch],
#             label_batch[:plot_per_batch],
#             pred_batch[:plot_per_batch],
#         ):
#             pred = torch.squeeze(pred).cpu().numpy()
#             label = torch.squeeze(label).cpu().numpy()
#             r = int(r.item())

#             assert len(label) == len(pred)
#             x = list(range(len(pred)))

#             num_samples = r
#             samp_x = [0] * num_samples
#             plt.rcParams["figure.figsize"] = [20, 4]
#             plt.scatter(x, pred, label="pred")
#             plt.scatter(x, label, label="y")
#             plt.scatter(list(range(num_samples)), samp_x, label="samp")
#             plt.legend()
#             plt.savefig(os.path.join(plot_dir, f"plot_{i}.png"))
#             plt.close()
