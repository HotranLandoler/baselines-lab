"""Experiment logging"""
import os.path
from datetime import datetime
from argparse import Namespace

import matplotlib.pyplot as plt

from evaluator import Evaluator

_PLOTS_PATH = "./plots/"


class Logger:
    """Log the experiment and save to file"""

    def __init__(self, settings: Namespace):
        super().__init__()
        self._model_name = settings.model.upper()
        self._dataset = settings.dataset
        self._settings = str(settings)

        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

    def save_to_file(self, evaluator: Evaluator, path: str):
        """Save current log to file"""
        with open(path, 'a', encoding="utf-8") as f:
            f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}]\n")
            f.write(f"Model: {self._model_name}\n")
            f.write(f"Dataset: {self._dataset}\n")
            f.write("Note: \n")
            f.write(f"Settings: {self._settings}\n")
            for metric, results in evaluator.metric_results.items():
                f.write(f"{metric}: {results}\n")
            for result in evaluator.get_final_results():
                f.write(result)
            f.write("\n")

    def add_record(self, train_loss: float, val_loss: float):
        """Add record (train_loss, val_loss) for an epoch."""
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)

    def plot_and_save(self, num_epochs: int):
        """Plot losses to figure and save to file."""
        epochs = range(num_epochs)

        fig, ax = plt.subplots()
        ax.plot(epochs, self._train_losses, label="Train loss")
        ax.plot(epochs, self._val_losses, label="Val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"Losses of {self._model_name} on {self._dataset}, {num_epochs} epochs")
        ax.legend()

        filename = f"{datetime.now():%Y-%m-%d-%H:%M:%S}.svg"
        fig.savefig(os.path.join(_PLOTS_PATH, filename))
