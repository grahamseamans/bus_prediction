import os
import random
import torch
from data_types import Data_Info, Model_Params
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
import numbers
from data import get_dataloaders

Batch = Tuple
mp = Model_Params()


# https://github.com/deepmind/optax/blob/master/examples/quick_start.ipynb


def main(_):
    train_loader, val_loader, test_loader, data_info = get_dataloaders(mp)

    def net_fn(batch: Batch) -> jnp.ndarray:
        (non_category, category, r) = batch
        # https://dm-haiku.readthedocs.io/en/latest/api.html#causal
        mlp = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(300),
                jax.nn.relu,
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(data_info.trip_length),
            ]
        )
        return mlp(non_category)

    net = hk.without_apply_rng(hk.transform(net_fn))
    xs, ys = next(iter(val_loader))
    params = net.init(jax.random.PRNGKey(42), xs)

    start_learning_rate = 1e-1

    scheduler = optax.exponential_decay(
        init_value=start_learning_rate, transition_steps=1000, decay_rate=0.99
    )

    gradient_transform = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        optax.scale(-1.0),
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    )

    # optimizer = optax.adam(1e-2)
    # opt_state = optimizer.init(params)

    opt_state = gradient_transform.init(params)

    def compute_loss(params, x, y):
        y_pred = net.apply(params, x)
        y = jnp.squeeze(y)
        y_pred = np.squeeze(y_pred)
        print(y_pred[0])
        print(y[0])
        loss = jnp.mean(optax.l2_loss(y_pred, y))
        return loss

    for epoch in range(mp.epochs):
        for xs, ys in train_loader:
            loss = compute_loss(params, xs, ys)
            print(loss)
            grads = jax.grad(compute_loss)(params, xs, ys)
            updates, opt_state = gradient_transform.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        for xs, ys in val_loader:
            loss = compute_loss(params, xs, ys)
            print(loss)


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
