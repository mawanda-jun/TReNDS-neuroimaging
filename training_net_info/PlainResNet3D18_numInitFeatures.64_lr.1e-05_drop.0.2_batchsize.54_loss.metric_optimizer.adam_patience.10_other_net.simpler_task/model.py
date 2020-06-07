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
    PlainResNet3D18, \
    PlainDenseNet3DCustom, \
    PlainResNet18Siamese, \
    PlainResNet18SiameseGRU, \
    VAERegularized, \
    VAERegularizedSiamese

from pytorchtools import EarlyStopping, TReNDSLoss, TReNDSMetric, SingleAccuracies, TReNDSLossVAE

from apex import amp, optimizers

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from typing import Dict, Union

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model:
    def __init__(self,
                 net_hyperparams: Dict[str, Union[str, int, float]],
                 train_params,
                 ):

        train_params_keys = [
            'base_lr',
            'max_lr',
            'lr',
            'lr_decay',
            'use_apex',
            'weight_decay',
            'optimizer_type',
            'network_type',
            'loss_type'
        ]
        self.base_lr, \
        self.max_lr, \
        self.lr, \
        self.lr_decay, \
        self.use_apex, \
        self.weight_decay, \
        optimizer_type, \
        network_type, \
        loss_type \
            = [train_params[k] for k in train_params_keys]

        self.loss, self.optimizer, self.net = self.__build_model(
            net_type=network_type,
            net_hyperparams=net_hyperparams,
            optimizer_type=optimizer_type,
            loss_type=loss_type,
        )

        self.net.to(DEVICE)

        self.net, self.optimizer = amp.initialize(
            self.net,
            self.optimizer,
            enabled=self.use_apex
        )

        # Prepare gradient clipping
        # for p in self.net.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, 0, clip_value))

        # Define metric, loss, optimizer
        self.metric = TReNDSMetric()  # Define metric function
        self.accuracies = SingleAccuracies()  # Define single percentage function
        self.metric.requires_grad = False  # Disable grad for function since useless
        self.accuracies.require_grad = False

    def __build_model(self,
                      net_type,
                      net_hyperparams,
                      optimizer_type,
                      loss_type
                      ) -> (nn.Module, nn.Module, torch.optim, nn.Module):
        # Mandatory parameters to be used.
        dropout_prob = net_hyperparams['dropout_prob']
        num_init_features = net_hyperparams['num_init_features']

        # Define custom network. In each one the specific parameters must be added from self.net_params
        if net_type == 'ShallowNet':
            network: nn.Module = ShallowNet(dropout_prob)
        # elif net_type == 'CustomDenseNet3D':
        #     network: nn.Module = CustomDenseNet3D(dropout_prob)
        elif net_type == 'CustomResNet3D10':
            network = CustomResNet3D10(dropout_prob, num_init_features)
        elif net_type == 'CustomResNet18Siamese':
            network = CustomResNet18Siamese(dropout_prob, num_init_features, self.use_apex)
        elif net_type == 'CustomResNet18SiameseMashup':
            network = CustomResNet18SiameseMashup(dropout_prob, num_init_features)
        elif net_type == 'PlainResNet18Siamese':
            network = PlainResNet18Siamese(dropout_prob, num_init_features, self.use_apex)
        elif net_type == 'PlainResNet18SiameseGRU':
            network = PlainResNet18SiameseGRU(dropout_prob, num_init_features)
        # elif net_type == 'CustomResNet3D50':
        #     network = CustomResNet3D50(dropout_prob)
        elif net_type == 'PlainDenseNet3DCustom':
            network = PlainDenseNet3DCustom(dropout_prob, num_init_features=num_init_features)
        elif net_type == 'PlainDenseNet3D121':
            network = PlainDenseNet3D121(dropout_prob, num_init_features=num_init_features)
        elif net_type == 'PlainDenseNet3D161':
            network = PlainDenseNet3D161(dropout_prob, num_init_features=num_init_features)
        elif net_type == 'PlainResNet3D10':
            network = PlainResNet3D10(dropout_prob, num_init_features=num_init_features)
        elif net_type == 'PlainResNet3D18':
            network = PlainResNet3D18(dropout_prob, num_init_features=num_init_features)
        elif net_type == 'PlainResNet3D50':
            network = PlainResNet3D50(dropout_prob, num_init_features=num_init_features, use_apex=self.use_apex)
        elif net_type == 'PlainResNet3D101':
            network = PlainResNet3D101(dropout_prob, num_init_features=num_init_features)
        elif net_type == 'VAERegularized':
            network = VAERegularized(dropout_prob, num_init_features)
        elif net_type == 'VAERegularizedSiamese':
            network = VAERegularizedSiamese(dropout_prob, num_init_features)
        else:
            raise ValueError("Bad network type. Please choose ShallowNet or ...")

        # Define loss
        if loss_type == 'humber':
            loss_fn = nn.SmoothL1Loss()  # Define loss_type function
        elif loss_type == 'metric':
            loss_fn = TReNDSLoss()
        elif loss_type == 'metric_vae':
            loss_fn = TReNDSLossVAE()
        elif loss_type == 'MAE':
            loss_fn = nn.L1Loss()
        elif loss_type == 'MSE':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("Bad loss type. Please choose humber or...")

        if optimizer_type == 'adam':
            # Define the optimizer. It wants to know which parameters are being optimized.
            optimizer_fn = torch.optim.Adam(params=network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_type == 'adamw':
            # Define the optimizer. It wants to know which parameters are being optimized.
            optimizer_fn = torch.optim.AdamW(params=network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_type == 'SGD':
            optimizer_fn = torch.optim.SGD(params=network.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay,
                                           nesterov=True)
        else:
            raise ValueError('Bad optimizer type. Please choose adam or ...')

        return loss_fn, optimizer_fn, network

    def __save(self, run_path, metric, epoch):
        state = {
            'state_dict': self.net.state_dict(),
            'optim_state': self.optimizer.state_dict()
        }
        if self.use_apex:
            state['apex_state'] = amp.state_dict()
        filename = 'ep_{}_checkpoint_{:.8f}.pt'.format(epoch, metric)
        filepath = os.path.join(run_path, filename)
        torch.save(state, filepath)

    def fit(self, epochs, train_loader, val_loader, patience, run_path=None, last_epoch=-1):
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        # cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_loader), 1e-8)
        on_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3,
                                                                          factor=0.5)
        # decreasing_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decay)
        cyclic_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.base_lr, max_lr=self.max_lr,
                                                                step_size_up=len(train_loader), cycle_momentum=False,
                                                                gamma=self.lr_decay, last_epoch=last_epoch)

        start_epoch = torch.cuda.Event(enable_timing=True)
        start_whole = torch.cuda.Event(enable_timing=True)
        end_whole = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)

        start_whole.record()

        whole_text = ''  # Keep record of history for further reference

        for i, epoch in enumerate(range(epochs)):
            if i > last_epoch:
                print("Training epoch {}".format(i))
                start_epoch.record()

                train_loss, train_metric, train_accuracies = self.net.train_batch(self.net, train_loader, self.loss,
                                                                                  self.metric, self.accuracies,
                                                                                  self.optimizer, cyclic_lr_scheduler,
                                                                                  DEVICE)
                val_loss, val_metric, val_accuracies = self.net.val_batch(self.net, val_loader, self.loss, self.metric,
                                                                          self.accuracies, DEVICE)

                end_epoch.record()
                torch.cuda.synchronize(DEVICE)
                # Calculate elapsed time
                elapsed_seconds = start_epoch.elapsed_time(
                    end_epoch) / 1000
                elapsed_minutes = elapsed_seconds // 60
                elapsed_seconds = round(elapsed_seconds % 60)
                space = "\n{}".format(''.join(["----" for _ in range(9)]))
                text = "\nEPOCH: {}\t\tElapsed_time: {:.0f}m{:.0f}s".format(epoch, elapsed_minutes, elapsed_seconds)
                text += "\n\t\t\t\tTrain\t\tValidation"
                text += "\nLoss:\t\t\t{:.4f}\t\t{:.4f}".format(train_loss, val_loss)
                text += "\nMetric:\t\t\t{:.4f}\t\t{:.4f}".format(train_metric, val_metric)
                text += space
                text += "\nSingle performance"
                text += "\nage:\t\t\t{:.2f}%\t\t{:.2f}%".format(train_accuracies[0], val_accuracies[0])
                text += "\ndomain1_var1:\t{:.2f}%\t\t{:.2f}%".format(train_accuracies[1], val_accuracies[1])
                text += "\ndomain1_var2:\t{:.2f}%\t\t{:.2f}%".format(train_accuracies[1], val_accuracies[1])
                text += "\ndomain2_var1:\t{:.2f}%\t\t{:.2f}%".format(train_accuracies[2], val_accuracies[2])
                text += "\ndomain2_var2:\t{:.2f}%\t\t{:.2f}%".format(train_accuracies[3], val_accuracies[3])
                text += space
                text += "\nLearning rate:\t{:.2e}".format(cyclic_lr_scheduler.get_last_lr()[0])
                text += space
                text += space

                print(text)
                whole_text += text

                # The if statement is not slowing down training since each epoch last very long.
                epoch_val_metric = val_metric
                epoch_train_metric = train_metric
                early_stopping(epoch_train_metric, epoch_val_metric, self.net)
                if early_stopping.save_checkpoint and run_path:
                    self.__save(run_path, epoch_val_metric, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                on_plateau_scheduler.step(val_metric)
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

        for ID, output in tqdm(zip(IDs, outputs), desc='Writing predictions on submission.csv file...',
                               total=len(outputs)):
            sub_names_part = [str(int(ID)) + sn for sn in sub_names]
            for name, out in zip(sub_names_part, output):
                submission.loc[len(submission['Id'])] = [name, out]
        submission.to_csv(os.path.join(run_path, 'submission.csv'), index=False)
