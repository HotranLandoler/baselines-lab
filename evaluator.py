import torch
import torchmetrics.functional.classification as metrics
from torch import Tensor
from typing import Literal


class Evaluator:
    """Evaluate experiment results with given metrics"""
    def __init__(self, use_metrics: list[Literal['AUC', 'AP']], num_classes: int):
        super(Evaluator, self).__init__()
        self.metric_results: dict[str, list] = {metric: [] for metric in use_metrics}
        self.__num_classes = num_classes

    def evaluate_test(self, predicts: Tensor, target: Tensor, test_mask: Tensor):
        """Evaluate on the test set"""
        for metric, results in self.metric_results.items():
            result = self._evaluate(metric, predicts, target, test_mask)
            results.append(result.item())
            yield f"{metric}: {result:.6f}, "

    def get_final_results(self):
        """Return final results by 'mean ± std'"""
        for metric, results in self.metric_results.items():
            final = torch.tensor(results)
            yield f"Final {metric}: {final.mean().item():.4f} ± {final.std(dim=0).item():.4f}, "

    def _evaluate(self,
                  metric_name: str,
                  predicts: Tensor,
                  target: Tensor,
                  mask: Tensor):
        if metric_name == 'AUC':
            return metrics.multiclass_auroc(predicts[mask], target[mask],
                                            num_classes=self.__num_classes)
        elif metric_name == 'AP':
            return metrics.multiclass_average_precision(
                predicts[mask], target[mask],
                num_classes=self.__num_classes)
        else:
            raise ValueError('Invalid metric name')

    def __str__(self):
        return f"Results: {self.metric_results}"
