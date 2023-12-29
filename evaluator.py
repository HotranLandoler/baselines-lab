import torch
import torchmetrics.functional.classification as metrics
from torch import Tensor
from typing import Literal


class Evaluator:
    """Evaluate experiment results with given metrics"""
    def __init__(self, use_metrics: list[Literal['AUC', 'AP']], num_classes: int, runs: int):
        super(Evaluator, self).__init__()
        self.metric_results: dict[str, list] = {metric: [0. for _ in range(runs)] for metric in use_metrics}
        self.__num_classes = num_classes

    def evaluate_test(self, run: int, predicts: Tensor, target: Tensor, test_mask: Tensor):
        """Evaluate on the test set"""
        for metric, results in self.metric_results.items():
            result = self._evaluate(metric, predicts, target, test_mask)
            # if results[run] > result.item():
            #     break

            results[run] = result.item()
            # print(f"{metric}-{run}: {result.item()}")

    def print_run_results(self, run: int):
        print(f"Run {run}: ", end='')
        for metric, results in self.metric_results.items():
            print(f"{metric}: {results[run]:.6f}, ", end='')
        print("")

    def get_final_results(self):
        """Return final results by 'mean ± std'"""
        for metric, results in self.metric_results.items():
            final = torch.tensor(results)
            # print(results)
            yield f"Final {metric}: {final.mean().item():.4f} ± {final.std(dim=0).item():.4f}, "

    def _evaluate(self,
                  metric_name: str,
                  predicts: Tensor,
                  target: Tensor,
                  mask: Tensor):
        match metric_name:
            case "AUC":
                return metrics.multiclass_auroc(predicts[mask], target[mask],
                                                num_classes=self.__num_classes)
            case "AP":
                return metrics.multiclass_average_precision(
                    predicts[mask], target[mask],
                    num_classes=self.__num_classes)
            case "Recall":
                return metrics.multiclass_recall(predicts[mask], target[mask],
                                                 num_classes=self.__num_classes)
            case _:
                raise ValueError('Invalid metric name')

    def __str__(self):
        return f"Results: {self.metric_results}"
