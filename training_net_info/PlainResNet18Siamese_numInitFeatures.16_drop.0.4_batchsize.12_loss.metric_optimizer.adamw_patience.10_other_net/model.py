from network import \
    ShallowNet, \
    CustomResNet18Siamese, \
    CustomResNet18SiameseMashup, \
    CustomResNet3D10, \
    PlainResNet3D50, \
    PlainResNet3D101, \
    PlainDenseNet3D121, \
    PlainDenseNet3D161, \
    PlainResNet3D10, \
    PlainDenseNet3DCustom, \
    PlainResNet18Siamese
from pytorchtools import EarlyStopping, TReNDSLoss, SingleAccuracies

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from typing import Dict, Union

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self,
                 net_type: str,
                 net_hyperparams: Dict[str, Union[str, int, float]],
                 optimizer_type: str = 'adam',
                 loss_type: str = 'metric',  # SmoothL1Loss
                 lr: float = .01,
                 lr_decay = .95
                 ):
        self.net_type = net_type
        self.net_hyperparams = net_hyperparams
        self.optimizer = optimizer_type
        self.loss = loss_type
        self.lr = lr
        self.lr_decay = lr_decay

        self.metric, self.loss, self.optimizer, self.net = self.__build_model()

        self.net.to(DEVICE)

    def __build_model(self) -> (nn.Module, nn.Module, torch.optim, nn.Module):
        # Mandatory parameters to be used.
        dropout_prob = self.net_hyperparams['dropout_prob']
        num_init_features = self.net_hyperparams['num_init_features']

        # Define custom network. In each one the specific parameters must be added from self.net_params
        if self.net_type == 'ShallowNet':
            network: nn.Module = ShallowNet(dropout_prob)
        # elif self.net_type == 'CustomDenseNet3D':
        #     network: nn.Module = CustomDenseNet3D(dropout_prob)
        elif self.net_type == 'CustomResNet3D10':
            network = CustomResNet3D10(dropout_prob, num_init_features)
        elif self.net_type == 'CustomResNet18Siamese':
            network = CustomResNet18Siamese(dropout_prob, num_init_features)
        elif self.net_type == 'CustomResNet18SiameseMashup':
            network = CustomResNet18SiameseMashup(dropout_prob, num_init_features)
        elif self.net_type == 'PlainResNet18Siamese':
            network = PlainResNet18Siamese(dropout_prob, num_init_features)
        # elif self.net_type == 'CustomResNet3D50':
        #     network = CustomResNet3D50(dropout_prob)
        elif self.net_type == 'PlainDenseNet3DCustom':
            network = PlainDenseNet3DCustom(dropout_prob, num_init_features=num_init_features)
        elif self.net_type == 'PlainDenseNet3D121':
            network = PlainDenseNet3D121(dropout_prob, num_init_features=num_init_features)
        elif self.net_type == 'PlainDenseNet3D161':
            network = PlainDenseNet3D161(dropout_prob, num_init_features=num_init_features)
        elif self.net_type == 'PlainResNet3D10':
            network = PlainResNet3D10(dropout_prob, num_init_features=num_init_features)
        elif self.net_type == 'PlainResNet3D50':
            network = PlainResNet3D50(dropout_prob, num_init_features=num_init_features)
        elif self.net_type == 'PlainResNet3D101':
            network = PlainResNet3D101(dropout_prob, num_init_features=num_init_features)
        else:
            raise ValueError("Bad network type. Please choose ShallowNet or ...")

        # Define metric, loss, optimizer
        metric_fn = SingleAccuracies()  # Define metric function
        metric_fn.requires_grad = False  # Disable grad for function since useless

        if self.loss == 'humber':
            loss_fn = nn.SmoothL1Loss()  # Define loss function
        elif self.loss == 'metric':
            loss_fn = TReNDSLoss()
        elif self.loss == 'MAE':
            loss_fn = nn.L1Loss()
        elif self.loss == 'MSE':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("Bad loss type. Please choose humber or...")

        if self.optimizer == 'adam':
            # Define the optimizer. It wants to know which parameters are being optimized.
            optimizer_fn = torch.optim.Adam(params=network.parameters(), lr=self.lr, weight_decay=1e-2)
        elif self.optimizer == 'adamw':
            # Define the optimizer. It wants to know which parameters are being optimized.
            optimizer_fn = torch.optim.AdamW(params=network.parameters(), lr=self.lr, weight_decay=1e-2)
        elif self.optimizer == 'SGD':
            optimizer_fn = torch.optim.SGD(params=network.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-7, nesterov=True)
        else:
            raise ValueError('Bad optimizer type. Please choose adam or ...')

        return metric_fn, loss_fn, optimizer_fn, network

    def __save(self, run_path, metric, epoch):
        state = {
            'state_dict': self.net.state_dict(),
            'optim_state': self.optimizer.state_dict()
        }
        filename = 'ep_{}_checkpoint_{:.8f}.pt'.format(epoch, metric)
        filepath = os.path.join(run_path, filename)
        torch.save(state, filepath)

    def fit(self, epochs, train_loader, val_loader, patience, run_path=None):
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        # cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_loader), 1e-8)
        on_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        # decreasing_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decay)
        cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=len(train_loader), cycle_momentum=False, gamma=self.lr_decay)

        start_epoch = torch.cuda.Event(enable_timing=True)
        start_whole = torch.cuda.Event(enable_timing=True)
        end_whole = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)

        start_whole.record()

        whole_text = ''  # Keep record of history for further reference

        for epoch in range(epochs):
            start_epoch.record()

            train_loss, train_metric = self.net.train_batch(self.net, train_loader, self.loss, self.metric, self.optimizer, cyclic_lr_scheduler, DEVICE)
            val_loss, val_metric = self.net.val_batch(self.net, val_loader, self.loss, self.metric, DEVICE)

            end_epoch.record()
            torch.cuda.synchronize(DEVICE)
            # Calculate elapsed time
            elapsed_seconds = start_epoch.elapsed_time(
                        end_epoch) / 1000
            elapsed_minutes = elapsed_seconds // 60
            elapsed_seconds = round(elapsed_seconds % 60)
            space = "\n{}".format(''.join(["----" for _ in range(9)]))
            text = "\nEPOCH: {}\t\tElapsed_time: {:.0f}m{:.0f}s".format(epoch + 1, elapsed_minutes, elapsed_seconds)
            text += "\n\t\t\t\tTrain\t\tValidation"
            text += "\nLoss:\t\t\t{:.4f}\t\t{:.4f}".format(train_loss, val_loss)
            text += space
            text += "\nACCURACY"
            text += "\nage:\t\t\t{:.2f}%\t\t{:.2f}%".format(train_metric[0], val_metric[0])
            text += "\ndomain1_var1:\t{:.2f}%\t\t{:.2f}%".format(train_metric[1], val_metric[1])
            text += "\ndomain1_var2:\t{:.2f}%\t\t{:.2f}%".format(train_metric[1], val_metric[1])
            text += "\ndomain2_var1:\t{:.2f}%\t\t{:.2f}%".format(train_metric[2], val_metric[2])
            text += "\ndomain2_var2:\t{:.2f}%\t\t{:.2f}%".format(train_metric[3], val_metric[3])
            text += space
            text += "\nLearning rate:\t{:.2e}".format(cyclic_lr_scheduler.get_last_lr()[0])
            text += space
            text += space

            print(text)
            whole_text += text
            # print(
            #     "\nEpoch: {}\ttrain metric: {:.4f} loss: {:.4f}\t\tval metric: {:.4f} loss: {:.4}\ttime: {:.0f}m{:.0f}s\t lr: {:.2e}".format(
            #         epoch+1,
            #         train_metric,
            #         train_loss,
            #         val_metric,
            #         val_loss,
            #         elapsed_minutes,
            #         elapsed_seconds,
            #         cyclic_lr_scheduler.get_last_lr()[0]
            #     ))
            # Update early stopping. This is really useful to stop training in time.
            # The if statement is not slowing down training since each epoch last very long.
            epoch_val_loss = val_loss.item()
            epoch_train_loss = train_loss.item()
            early_stopping(epoch_train_loss, epoch_val_loss, self.net)
            if early_stopping.save_checkpoint and run_path:
                self.__save(run_path, epoch_val_loss, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            on_plateau_scheduler.step(val_loss)
            # decreasing_lr_scheduler.step()

            # Save history to file
            open(os.path.join(run_path, 'history.txt'), 'w').write(whole_text)
        end_whole.record()
        torch.cuda.synchronize(DEVICE)
        print("Elapsed time: {:.4f}m".format(start_whole.elapsed_time(end_whole) / 60000))

        # Return the best metric that we register for early stopping.
        return early_stopping.val_metric_min

    def submit(self, test_loader, run_path):
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
