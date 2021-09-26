cat_names = ['direction', 'door', 'lift', 'location_id', 'schedule_status', 'service_key', 'train', 'vehicle_number']
non_cat_names = ['dwell', 'estimated_load', 'maximum_speed', 'offs', 'ons', 'pattern_distance', 'train_mileage', 'arrive_deviance_departure_delta']
label_names = ['arrival_deviance']
time_names = ['stop_timestamp']
cardinality = [1, 2, 9, 87, 6, 4, 28, 376]
recompute=False
trip_direction=1
trip_length = 83
learning_rate = 0.001
embedding_scale = 4
kernel_size = 16
dilation = 2
conv_channels = 16
conv_stride = 16
batch_size = 8
epochs = 50


# MOGAMI_GAWA

# can't save as a key, anything that starts with __ or have the key be a module type...
# you could also replace the isinstance moduletype with not allowoing config, and os

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
        f.write("\n\n")
        f.write(split)
        f.write(funcs)
