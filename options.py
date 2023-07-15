"""Prepare experiment settings"""
import tomli
from argparse import ArgumentParser
from argparse import Namespace
from typing import Literal, Any

_CONFIG_PATH = "config.toml"


def prepare_args():
    """Build a parser and prepare experiment args"""
    parser = ArgumentParser(description="Prepare args")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--loss-weight', type=int, default=50)
    parser.add_argument('--save-log', action='store_true', help='save experiment log to file')
    parser.add_argument('--plot', action='store_true', help='plot losses and save to file')
    args = parser.parse_args()
    _parse_configs(args)
    return args


def _parse_configs(args: Namespace):
    """Parse the toml config file and add to args"""
    with open(_CONFIG_PATH, "rb") as file:
        configs = tomli.load(file)
    # Add model/dataset specific args
    _add_specific_args(configs, args, prefix="models", key="model")
    _add_specific_args(configs, args, prefix="datasets", key="dataset")
    # Add rest of the args
    _add_dict_to_args(configs, args)


def _add_specific_args(configs: dict[str, Any],
                       args: Namespace,
                       prefix: Literal["models", "datasets"],
                       key: Literal["model", "dataset"]):
    """Add specific configs to args based on key."""
    model_configs: dict = configs.pop(prefix)
    _add_dict_to_args(model_configs[configs[key]], args)


def _add_dict_to_args(source: dict, args: Namespace):
    for key, value in source.items():
        args.__dict__[key] = value
