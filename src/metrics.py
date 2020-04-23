from pdb import set_trace
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator(object):

    def __init__(self, metrics) -> None:
        super().__init__()
        self.metrics = metrics

    def _format_tensor(self, tensor):
        return tensor.cpu().detach().view(-1, )

    def _mean_metrics(self, logs, key):
        return logs[key] if isinstance(logs, dict) else torch.stack([item[key] for item in logs]).mean()

    def accuracy(self, true, pred):
        return torch.tensor(accuracy_score(self._format_tensor(true), self._format_tensor(pred)))

    def precision(self, true, pred):
        return torch.tensor(precision_score(self._format_tensor(true), self._format_tensor(pred), average="macro"))

    def recall(self, true, pred):
        return torch.tensor(recall_score(self._format_tensor(true), self._format_tensor(pred), average="macro"))

    def f1(self, true, pred):
        return torch.tensor(f1_score(self._format_tensor(true), self._format_tensor(pred), average="macro"))

    def generate_logs(self, loss, preds, true, prefix):
        return {f"{prefix}_loss": loss,
                **{f"{prefix}_{key}": MetricsCalculator.__dict__[key](self, true, preds) for key in self.metrics}}

    def generate_mean_metrics(self, outputs, prefix):
        mean_keys = ["loss"] + self.metrics
        return {f"{prefix}_{key}": self._mean_metrics(outputs, f"{prefix}_{key}") for key in mean_keys}
