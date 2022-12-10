"""Experiment logging"""
from datetime import datetime
from argparse import Namespace

from evaluator import Evaluator


class Logger:
    """Log the experiment and save to file"""

    def __init__(self, settings: Namespace):
        super().__init__()
        self.__settings = str(settings)

    def save_to_file(self, evaluator: Evaluator, path: str):
        """Save current log to file"""
        with open(path, 'a', encoding="utf-8") as f:
            f.write(f"\n\n[{datetime.now():%Y-%m-%d %H:%M:%S}]\n")
            f.write(f"Settings: {self.__settings}\n")
            for metric, results in evaluator.metric_results.items():
                f.write(f"{metric}: {results}\n")
            for result in evaluator.get_final_results():
                f.write(result)
