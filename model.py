from network import ShallowNet, train_batch, val_batch
from pytorchtools import EarlyStopping, TReNDSMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Union

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self,
                 net_type: str,
                 net_params: Dict[str, Union[str, int]],
                 optimizer: str = 'adam',
                 loss: nn.Module = 'humber', # SmoothL1Loss
                 lr: float = .01,
                 ):
        self.net_type = net_type
        self.net_params = net_params
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr

        self.metric_fn, self.loss_fn, self.optimizer_fn, self.net = self.__build_model()

    def __build_model(self) -> (nn.Module, nn.Module, torch.optim, nn.Module):
        # Mandatory parameters to be used.
        dropout_prob = self.net_params['dropout_prob']
        fnc_dim = self.net_params['fnc_dim']
        sbm_dim = self.net_params['sbm_dim']

        # Define custom network. In each one the specific parameters must be added from self.net_params
        if self.net_type == 'ShallowNet':
            network: nn.Module = ShallowNet(fnc_dim, sbm_dim, dropout_prob)
        else:
            raise ValueError("Bad network type. Please choose ShallowNet or ...")

        network.to(DEVICE)  # Send network module to DEVICE

        # Define metric, loss, optimizer
        metric_fn = TReNDSMetrics()  # Define metric function

        if self.loss == 'humber':
            loss_fn = nn.SmoothL1Loss()  # Define loss function
        elif self.loss == 'metric':
            loss_fn = TReNDSMetrics()
        else:
            raise ValueError("Bad loss type. Please choose humber or...")

        if self.optimizer == 'adam':
            # Define the optimizer. It wants to know which parameters are being optimized.
            optimizer_fn = torch.optim.Adam(params=network.parameters(), lr=self.lr, weight_decay=1e-5)
        else:
            raise ValueError('Bad optimizer type. Please choose adam or ...')

        return metric_fn, loss_fn, optimizer_fn, network

    def fit(self, epochs, train_loader, val_loader, patience, run_name = None):
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_fn, len(train_loader), 1e-7)

        start_epoch = torch.cuda.Event(enable_timing=True)
        start_whole = torch.cuda.Event(enable_timing=True)
        end_whole = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)

        start_whole.record()

        for epoch in range(epochs):
            start_epoch.record()

            train_loss, train_metric = train_batch(self.net, train_loader, self.loss_fn, self.metric_fn, self.optimizer_fn, DEVICE)
            val_loss, val_metric = val_batch(self.net, val_loader, self.loss_fn, self.metric_fn, DEVICE)

            end_epoch.record()
            torch.cuda.synchronize(DEVICE)
            print(
                "Epoch: {}\ttrain: acc: {:.4f} loss: {:.4f}\t\tval: acc: {:.4f} loss: {:.4f}\ttime: {:.4}s".format(
                    epoch,
                    train_metric,
                    train_loss,
                    val_metric,
                    val_loss,
                    start_epoch.elapsed_time(
                        end_epoch) / 1000))
            # Update early stopping. This is really useful to stop training in time.
            # The if statement is not slowing down training since each epoch last very long.
            # PLEASE TAKE NOTE THAT we are using epoch_val_acc, since it brings the score function of the competition
            float_epoch_val_acc = val_loss.detach().cpu().numpy()
            float_epoch_train_acc = train_loss.detach().cpu().numpy()
            early_stopping(float_epoch_train_acc, float_epoch_val_acc, self.net)
            # if early_stopping.save_checkpoint and run_name:
            #     self.save(run_name, float_epoch_val_acc, epoch)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            # Update scheduler
            scheduler.step()
        end_whole.record()
        torch.cuda.synchronize(DEVICE)
        print("Elapsed time: {:.4f}s".format(start_whole.elapsed_time(end_whole) / 1000))

        # Return val_loss_min for KFold - which is, the metric that we register for early stopping.
        return early_stopping.val_loss_min
