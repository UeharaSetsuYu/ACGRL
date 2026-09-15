from copy import deepcopy


_CONFIGS = {
    "NGs": {
        "view_num": 3,
        "batch_size": 64,
        "architectures": [
            [2000, 1024, 1024, 1024, 128],
            [2000, 1024, 1024, 1024, 128],
            [2000, 1024, 1024, 1024, 128],
        ],
    },
    "BBCSport": {
        "view_num": 2,
        "batch_size": 256,
        "architectures": [
            [3183, 1024, 1024, 1024, 128],
            [3203, 1024, 1024, 1024, 128],
        ],
    },
    "Hdigit": {
        "view_num": 2,
        "batch_size": 256,
        "architectures": [
            [784, 1024, 1024, 1024, 128],
            [256, 1024, 1024, 1024, 128],
        ],
    },
    "Cora": {
        "view_num": 2,
        "batch_size": 256,
        "architectures": [
            [2708, 1024, 1024, 1024, 128],
            [1433, 1024, 1024, 1024, 128],
        ],
    },
}


def get_default_config(dataset_name):
    try:
        return deepcopy(_CONFIGS[dataset_name])
    except KeyError as exc:
        supported = ", ".join(_CONFIGS)
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from: {supported}.") from exc
