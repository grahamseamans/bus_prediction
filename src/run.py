from matplotlib import pyplot as plt
from data_read_parse import get_data
from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Tuple
from data import get_dataloaders
from tqdm import tqdm
import config

Batch = Tuple


class GLU_Conv(hk.Module):
    # def __init__(self, mp: Model_Params):
    #     super(CustomModule, self).__init__()

    def __call__(self, X):
        x_1 = hk.Conv1D(
            config.conv_channels,
            config.kernel_size,
            config.conv_stride,
            config.dilation,
            hk.pad.causal(config.kernel_size),
        )(X)
        x_2 = hk.Conv1D(
            config.conv_channels,
            config.kernel_size,
            config.conv_stride,
            config.dilation,
            hk.pad.causal(config.kernel_size),
        )(X)
        x_2 = jax.nn.sigmoid(x_2)
        return x_1 * x_2


class CustomModule(hk.Module):
    # def __init__(self, data_info):
    #     super(CustomModule, self).__init__()
    #     self.data_info = data_info

    def __call__(self, X, is_training):
        (non_category, category, r) = X
        x = hk.BatchNorm(True, True, decay_rate=0.9)(non_category, is_training)
        x = GLU_Conv(x)
        x = GLU_Conv(x)
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.elu(x)
        x = hk.Linear(config.trip_length)(x)
        x = jnp.expand_dims(x, -1)
        return x


def main(_):
    train_loader, val_loader, test_loader, data_info = get_dataloaders()

    # start_learning_rate = 1e-1
    # scheduler = optax.exponential_decay(
    #     init_value=start_learning_rate, transition_steps=1000, decay_rate=0.99
    # )
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        # optax.scale_by_schedule(scheduler),
        optax.scale(config.learning_rate),
        optax.scale(-1.0),
    )

    def _forward(X, is_training: bool) -> jnp.ndarray:
        net = CustomModule()
        return net(X, is_training)

    forward = hk.without_apply_rng(hk.transform_with_state(_forward))
    Xs, ys = next(iter(val_loader))
    params, state = forward.init(jax.random.PRNGKey(42), data_info, Xs, True)
    opt_state = gradient_transform.init(params)

    def loss(params, state, X, y):
        y_pred, state = forward.apply(params, state, data_info, X, is_training=True)
        return jnp.mean(optax.huber_loss(y_pred, y))

    @jax.jit
    def train(params, state, opt_state, Xs, ys):
        grads = jax.grad(loss)(params, state, Xs, ys)
        updates, opt_state = gradient_transform.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, state, opt_state

    @jax.jit
    def metric(params, state, X, y):
        y_pred, state = forward.apply(params, state, data_info, X, False)
        return jnp.mean(optax.huber_loss(y_pred, y))

    def prog_bar(iterable, desc):
        return tqdm(
            iterable, total=len(iterable), desc=desc, unit="batch", mininterval=0.5
        )

    for epoch in range(config.epochs):
        train_loss = 0
        for Xs, ys in prog_bar(train_loader, "training"):
            params, state, opt_state = train(params, state, opt_state, Xs, ys)
            train_loss += metric(params, state, Xs, ys)
        val_loss = 0
        for Xs, ys in prog_bar(val_loader, "validation"):
            val_loss += metric(params, state, Xs, ys)
        print(
            f"epoch: {epoch} train loss avg is {train_loss / len(train_loader):.2f} and val loss avg is {val_loss / len(val_loader):.2f}"
        )
        # bar.update(1)
        # bar.set_postfix({"loss": f"{val_loss / batch_idx:.2f}"})
        # with prog_bar(train_loader, desc="training") as bar:
        # for batch_idx, (Xs, ys) in prog_bar(enumerate(train_loader)):


if __name__ == "__main__":
    app.run(main)
    # main("banana")

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
