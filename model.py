from network import ShallowNet, SmartDense3D
from pytorchtools import EarlyStopping, TReNDSMetrics

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Union
import gc

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self,
                 net_type: str,
                 net_hyperparams: Dict[str, Union[str, int, float]],
                 optimizer: str = 'adam',
                 loss: str = 'humber',  # SmoothL1Loss
                 lr: float = .01,
                 ):
        self.net_type = net_type
        self.net_hyperparams = net_hyperparams
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr

        self.metric_fn, self.loss_fn, self.optimizer_fn, self.net = self.__build_model()

    def __build_model(self) -> (nn.Module, nn.Module, torch.optim, nn.Module):
        # Mandatory parameters to be used.
        dropout_prob = self.net_hyperparams['dropout_prob']

        # Define custom network. In each one the specific parameters must be added from self.net_params
        if self.net_type == 'ShallowNet':
            network: nn.Module = ShallowNet(dropout_prob)
        elif self.net_type == 'SmartDense3D':
            network: nn.Module = SmartDense3D(dropout_prob)
        else:
            raise ValueError("Bad network type. Please choose ShallowNet or ...")

        # Define metric, loss, optimizer
        metric_fn = TReNDSMetrics()  # Define metric function

        if self.loss == 'humber':
            loss_fn = nn.SmoothL1Loss()  # Define loss function
        elif self.loss == 'metric':
            loss_fn = TReNDSMetrics()
        elif self.loss == 'MAE':
            loss_fn = nn.L1Loss()
        elif self.loss == 'MSE':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("Bad loss type. Please choose humber or...")

        if self.optimizer == 'adam':
            # Define the optimizer. It wants to know which parameters are being optimized.
            optimizer_fn = torch.optim.Adam(params=network.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.optimizer == 'SGD':
            optimizer_fn = torch.optim.SGD(params=network.parameters(), lr=self.lr, momentum=0.3, weight_decay=1e-7)
        else:
            raise ValueError('Bad optimizer type. Please choose adam or ...')

        return metric_fn, loss_fn, optimizer_fn, network

    def __save(self, run_path, metric, epoch):
        state = {
            'state_dict': self.net.state_dict()
        }
        filepath = os.path.join(run_path, 'checkpoint_' + str(metric) + '_ep' + str(epoch) + '.pt')
        torch.save(state, filepath)

    def fit(self, epochs, train_loader, val_loader, patience, run_path=None):
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_fn, len(train_loader), 1e-7)

        start_epoch = torch.cuda.Event(enable_timing=True)
        start_whole = torch.cuda.Event(enable_timing=True)
        end_whole = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)

        start_whole.record()

        for epoch in range(epochs):
            start_epoch.record()

            train_loss, train_metric = self.net.train_batch(self.net, train_loader, self.loss_fn, self.metric_fn, self.optimizer_fn, DEVICE)
            val_loss, val_metric = self.net.val_batch(self.net, val_loader, self.loss_fn, self.metric_fn, DEVICE)

            end_epoch.record()
            torch.cuda.synchronize(DEVICE)
            # Calculate elapsed time
            elapsed_seconds = start_epoch.elapsed_time(
                        end_epoch) / 1000
            elapsed_minutes = elapsed_seconds // 60
            elapsed_seconds = round(elapsed_seconds % 60)
            print(
                "\nEpoch: {}\ttrain metric: {:.4f} loss: {:.4f}\t\tval metric: {:.4f} loss: {:.4}\ttime: {:.0f}m{:.0f}s".format(
                    epoch+1,
                    train_metric,
                    train_loss,
                    val_metric,
                    val_loss,
                    elapsed_minutes,
                    elapsed_seconds
                ))
            # Update early stopping. This is really useful to stop training in time.
            # The if statement is not slowing down training since each epoch last very long.
            epoch_val_metric = val_metric.item()
            epoch_train_metric = train_metric.item()
            early_stopping(epoch_train_metric, epoch_val_metric, self.net)
            if early_stopping.save_checkpoint and run_path:
                self.__save(run_path, epoch_val_metric, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Update scheduler
            scheduler.step()
        end_whole.record()
        torch.cuda.synchronize(DEVICE)
        print("Elapsed time: {:.4f}m".format(start_whole.elapsed_time(end_whole) / 60000))

        # Return the best metric that we register for early stopping.
        return early_stopping.val_metric_min

    def submit(self, test_loader, run_path):
        print('Predicting test set...')
        IDs, outputs = self.net.predict_batch(self.net, test_loader, DEVICE)
        submission = pd.DataFrame(columns=['Id', 'Predicted'])
        sub_names = [
            '_age',
            '_domain1_var1',
            '_domain1_var2',
            '_domain2_var1',
            '_domain2_var2'
        ]

        for ID, output in tqdm(zip(IDs, outputs), desc='Writing predictions on submission.csv file...', total=len(outputs)):
            sub_names_part = [str(int(ID)) + sn for sn in sub_names]
            for name, out in zip(sub_names_part, output):
                submission.loc[len(submission['Id'])] = [name, out]
        submission.to_csv(os.path.join(run_path, 'submission.csv'), index=False)
