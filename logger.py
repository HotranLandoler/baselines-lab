"""Experiment logging"""
import os.path
from datetime import datetime
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor
from sklearn.manifold import TSNE

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

        filename = f"{datetime.now():%Y-%m-%d-%H_%M_%S}.svg"
        fig.savefig(os.path.join(_PLOTS_PATH, filename))

    def plot_embedding_visualization(self, embedding_test: Tensor,
                                     y_test: Tensor):
        """Visualize embeddings using t-SNE and save to file."""
        num_samples = 800

        embedding_test = embedding_test.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()

        embedding_reduced = TSNE(n_components=2, learning_rate='auto',
                                 init='random', perplexity=3).fit_transform(embedding_test)

        embedding_fraud = embedding_reduced[y_test == 1]
        embedding_benign = embedding_reduced[y_test == 0]

        embedding_fraud_sampled = embedding_fraud[
            np.random.randint(low=0, high=embedding_fraud.shape[0], size=num_samples)
        ]
        embedding_benign_sampled = embedding_benign[
            np.random.randint(low=0, high=embedding_benign.shape[0], size=num_samples)
        ]
        data_benign = np.concatenate((embedding_benign_sampled,
                                      np.zeros((num_samples, 1), dtype=int)), axis=-1)
        data_fraud = np.concatenate((embedding_fraud_sampled,
                                     np.ones((num_samples, 1), dtype=int)), axis=-1)

        column_titles = ["x1", "x2", "y"]
        df = pd.DataFrame(np.concatenate((data_benign, data_fraud), axis=0),
                          columns=column_titles)
        plot = sns.scatterplot(df, x="x1", y="x2", hue="y", legend=False)
        filename = f"tSNE-{self._model_name}-{self._dataset}.svg"
        plot.get_figure().savefig(os.path.join(_PLOTS_PATH, filename))
