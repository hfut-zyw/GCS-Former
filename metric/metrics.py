import abc

import torch


class Metric:
    """
    All parent classes for evaluation must implement the following 3 methods.
    """

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def accumulate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    def score(self):
        return self.accumulate()


class Accuracy(Metric):
    def __init__(self):
        self.num_correct = 0
        self.num_count = 0

    def update(self, pred, target):
        preds = pred.argmax(dim=-1)
        batch_correct = (preds == target).sum()
        batch_count = len(target)

        self.num_correct += batch_correct
        self.num_count += batch_count

    def accumulate(self):
        return (self.num_correct / self.num_count).item()

    def reset(self):
        self.num_correct = 0
        self.num_count = 0


class Accuracy2():
    def __init__(self):
        self.num_correct = 0
        self.num_count = 0

    def update(self, logits, labels):
        preds = logits.argmax(dim=-1)
        batch_correct = (preds[:, 1] == labels[:, 1]).sum()
        batch_count = len(labels)

        self.num_correct += batch_correct
        self.num_count += batch_count

    def accumulate(self):
        return (self.num_correct / self.num_count).item()

    def reset(self):
        self.num_correct = 0
        self.num_count = 0


if __name__ == "__main__":
    predicts = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
    labels = torch.tensor([0, 1])
    metric = Accuracy()
    metric.update(predicts, labels)
    print(metric.accumulate())
