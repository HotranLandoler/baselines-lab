"""Experiment logging"""
from datetime import datetime
from argparse import Namespace

from evaluator import Evaluator


class Logger:
    """Log the experiment and save to file"""

    def __init__(self, settings: Namespace):
        super().__init__()
        self._model_name = settings.model.upper()
        self._dataset = settings.dataset
        self._settings = str(settings)

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
