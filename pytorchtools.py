import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TReNDSMetrics(torch.nn.Module):
    """
    DeepFake competition metric: log-loss
    """

    def __init__(self):
        super(TReNDSMetrics, self).__init__()
        self.weights = torch.tensor([.3, .175, .175, .175, .175], dtype=torch.float64, device=DEVICE)

    def __loss(self, output, target):
        return torch.sum(self.weights * torch.sum(torch.abs(output-target)) / torch.sum(target))

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
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_checkpoint = False

    def __call__(self, train_loss, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint = True
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta and train_loss < val_loss:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            self.save_checkpoint = False
        else:
            self.best_score = score
            self.save_checkpoint = True
            self.val_loss_min = val_loss
            self.counter = 0
