import os

import numpy as np


class EarlyStopping:
    """Stop training when validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience: How long to wait after last time validation loss improved.
            verbose: If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min: float = np.inf

    def check_stop(self, val_loss: float) -> bool:
        """"""
        if val_loss >= self.val_loss_min:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.val_loss_min = val_loss
            self.counter = 0
            return False
