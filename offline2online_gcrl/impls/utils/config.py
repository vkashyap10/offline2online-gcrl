import yaml
from ml_collections import ConfigDict


def load_config(path: str) -> ConfigDict:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return ConfigDict(cfg_dict)
