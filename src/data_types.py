from dataclasses import dataclass, field


def_list = field(default_factory=list)


@dataclass
class Data_Info:
    cat_names: list = def_list
    non_cat_names: list = def_list
    label_names: list = def_list
    time_names: list = def_list
    cardinality: list = def_list
    trip_length: int = 0


@dataclass
class Model_Params:
    learning_rate: float = 1e-3
    embedding_scale: int = 4
    kernel_size: int = 8
    conv_channels: int = 16
    batch_size: int = 16
    epochs: int = 2
