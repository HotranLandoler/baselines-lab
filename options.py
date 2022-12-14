"""Prepare experiment settings"""
import tomli
from argparse import ArgumentParser
from argparse import Namespace

CONFIG_PATH = "config.toml"


def prepare_args():
    """Build a parser and prepare experiment args"""
    parser = ArgumentParser(description="Prepare args")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--loss-weight', type=int, default=50)
    parser.add_argument('--save-log', action='store_true', help='save experiment log to file')
    args = parser.parse_args()
    __parse_configs(args)
    return args


def __parse_configs(args: Namespace):
    """Parse the toml config file and add to args"""
    with open(CONFIG_PATH, "rb") as file:
        configs = tomli.load(file)
    model_configs: dict = configs.pop('models')
    # Add model specific args
    __add_dict_to_args(model_configs[configs['model']], args)
    # Add rest of the args
    __add_dict_to_args(configs, args)


def __add_dict_to_args(source: dict, args: Namespace):
    for key, value in source.items():
        args.__dict__[key] = value
