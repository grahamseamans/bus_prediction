cat_names = []
non_cat_names = []
label_names = []
time_names = []
cardinality = []
trip_length = 0
learning_rate = 0.001
embedding_scale = 4
kernel_size = 16
dilation = 2
conv_channels = 16
conv_stride = 16
batch_size = 8
epochs = 50
test = slice(2, 2, None)


# MOGAMI_GAWA

# can't save as a key, anything that starts with __ or have the key be a module type...

import config
import os
import types


def save():
    split = "# MOGAMI_GAWA"
    vars = {
        key: value
        for key, value in config.__dict__.items()
        if not (
            key.startswith("__") or isinstance(value, types.ModuleType) or key == "save"
        )
    }
    with open(os.path.join(os.getcwd(), "src", "config.py"), "r") as f:
        config_file = f.read()
        funcs = config_file.split(split, 1)[1]
    with open(os.path.join(os.getcwd(), "src", "config.py"), "w") as f:
        for key, value in vars.items():
            f.write(f"{key} = {repr(value)}\n")  # repr accounts for strings...
            print(key, value)
            print([type(x) for x in [key, value]])
        f.write("\n\n")
        f.write(split)
        f.write(funcs)
