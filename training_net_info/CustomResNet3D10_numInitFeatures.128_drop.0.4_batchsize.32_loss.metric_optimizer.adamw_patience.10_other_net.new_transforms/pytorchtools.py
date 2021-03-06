import numpy as np
import torch


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SingleAccuracies(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.weights = torch.tensor([.3, .175, .175, .175, .175], dtype=torch.float32, device=DEVICE)

    def __metric(self, output, target):
        nom = torch.sum(torch.abs(output-target), dim=0)
        denom = torch.sum(target, dim=0)
        return nom / denom

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        return (1. - self.__metric(output, target)) * 100


class TReNDSLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([.3, .175, .175, .175, .175], dtype=torch.float32, device=DEVICE)

    def __loss(self, output, target):
        nom = torch.sum(torch.abs(output-target), dim=0)
        denom = torch.sum(target, dim=0)
        return torch.sum(self.weights * nom / denom)

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        return self.__loss(output, target)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience"""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_min = np.Inf  # The lower the better
        self.delta = delta
        self.save_checkpoint = False

    def __call__(self, train_metric, val_metric, model=None):
        # Metric: the lower the better -> score: the higher the better
        val_score = - val_metric
        train_score = - train_metric

        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint = True
            self.val_metric_min = val_metric
        if val_score < self.best_score + self.delta and train_score > val_score + self.delta:  # apply patience only if train is better than val scores
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            self.save_checkpoint = False
        else:
            self.best_score = val_score
            self.save_checkpoint = True
            self.val_metric_min = val_metric
            self.counter = 0
