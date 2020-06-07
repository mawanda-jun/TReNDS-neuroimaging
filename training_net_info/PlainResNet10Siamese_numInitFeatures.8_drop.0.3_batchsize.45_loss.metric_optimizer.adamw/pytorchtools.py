import numpy as np
import torch
from losses import VAELoss


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TReNDSMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([.3, .175, .175, .175, .175], dtype=torch.float32, device=DEVICE)

    def _loss(self, output, target):
        nom = torch.sum(torch.abs(output-target), dim=0)
        denom = torch.sum(target, dim=0)
        return nom / denom

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return torch.sum(self.weights * self._loss(output, target))


class SingleAccuracies(TReNDSMetric):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        return (1. - self._loss(output, target)) * 100


class TReNDSLoss(TReNDSMetric):
    def __init__(self):
        super().__init__()
        # self.weights = torch.tensor([.4, .17, .17, .17, .19], dtype=torch.float32, device=DEVICE)


class TReNDSLossVAE(torch.nn.Module):
    def __init__(self):
        super(TReNDSLossVAE, self).__init__()
        self.metric = TReNDSLoss()
        self.vae = VAELoss(weight_KL=.1, weight_L2=.1)

    def forward(self, outputs, labels):
        out_features, reconstructed_image, input_image, z_mean, z_var = outputs
        return self.metric(out_features, labels) + self.vae(reconstructed_image, input_image, z_mean, z_var)


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
        elif val_score > self.best_score:
            self.best_score = val_score
            self.save_checkpoint = True
            self.val_metric_min = val_metric
            self.counter = 0
        else:
            self.save_checkpoint = False
