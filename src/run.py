import os
from data_types import Model_Params
from matplotlib import pyplot as plt
from data_read_parse import get_data
from absl import app
import numpy as np
from typing import Tuple
from data import get_dataloaders
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from tensorflow.python.framework.ops import disable_eager_execution

Batch = Tuple
mp = Model_Params()


def main():
    train, val, test, data_info = get_dataloaders(mp)

    x_shapes, y_shapes = next(iter(train))
    x_shapes = [x.shape[1:] for x in x_shapes]
    y_shapes = [y.shape[1:] for y in y_shapes]

    ''' (non_category, category, r), label '''

    non_category_input = keras.Input(shape=x_shapes[0])
    category_input = keras.Input(shape=x_shapes[1])
    r_input = keras.Input(shape=x_shapes[2])
    x = layers.BatchNormalization()(non_category_input)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='elu')(x)
    x = layers.Dense(data_info.trip_length)(x)
    output = x
    model = keras.Model(inputs=[non_category_input, category_input, r_input],
                        outputs=[output])

    keras.utils.plot_model(model, "model.png", show_shapes=True)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.Huber(),
        metrics='mae',
    )

    history = model.fit(train,
                        validation_data=val,
                        epochs=1,
                        workers=8,
                        use_multiprocessing=True)

    Xs, ys = next(iter(test)) 
    print(Xs)
    preds = model.predict_on_batch(Xs)
    for i, (X, pred, y) in enumerate(zip(Xs, preds, ys)):
        print(X)
        non_cats, cats, r = X
        plt.rcParams["figure.figsize"] = [20, 4]
        plt.scatter(x, pred, label="pred")
        plt.scatter(x, y, label="y")
        plt.scatter(list(range(r)), [0] * r, label="samp")
        plt.legend()
        plt.savefig(
            os.path.join(os.getcwd(), "data", "plots", f"plot_{i}.png"))
        plt.close()


if __name__ == "__main__":
    main()

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
