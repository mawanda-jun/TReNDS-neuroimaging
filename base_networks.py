import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from apex import amp


class BaseNetwork(nn.Module):
    """
    Base class to define common methods among all networks
    """
    def __init__(self, use_apex):
        # inizializzazione classe base - si fa sempre
        super().__init__()
        self.use_apex = use_apex

    def forward(self, inputs):
        pass

    @staticmethod
    def get_network_inputs(batch, DEVICE):
        return [b.to(DEVICE) for b in batch[0]]

    def train_batch(self, train_loader, loss_fn, metric_fn, accuracies_fn, optimizer, scheduler, DEVICE) -> (torch.Tensor, torch.Tensor):
        """
        Define training method only once. The only method that must be done is how the training gets the training inputs
        :param net:
        :param train_loader:
        :param loss_fn:
        :param metric_fn:
        :param optimizer:
        :param scheduler: scheduler that must be updated at every batch iteration
        :param DEVICE:
        :return:
        """
        self.to(DEVICE)
        self.train()
        running_loss = 0
        running_metric = 0
        running_acc = np.zeros(5)
        for batch in tqdm(train_loader, desc='Training...'):
            net_input = self.get_network_inputs(batch, DEVICE)
            labels = batch[1].to(DEVICE)

            # forward pass
            net_output = self.forward(net_input)

            del net_input

            # update networks
            loss = loss_fn(net_output, labels)
            metric = metric_fn(net_output, labels)
            accuracies = accuracies_fn(net_output, labels)

            del net_output

            # clear previous recorded gradients
            optimizer.zero_grad()

            # backward pass
            if self.use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)

            # update optimizer
            optimizer.step()

            running_loss += loss.item()
            running_metric += metric.item()
            running_acc += accuracies.detach().cpu().numpy()

            del loss
            del metric

            # Update scheduler
            if scheduler:
                scheduler.step()
            # else:
            #     break

        return running_loss / len(train_loader), running_metric / len(train_loader), running_acc / len(train_loader)

    def val_batch(self, val_loader, loss_fn, metric_fn, accuracies_fn, DEVICE) -> (torch.Tensor, torch.Tensor):
        self.to(DEVICE)
        self.eval()
        running_loss = 0
        running_metric = 0
        running_acc = np.zeros(5)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating...'):
                net_input = self.get_network_inputs(batch, DEVICE)
                labels = batch[1].to(DEVICE)

                # evaluate the network over the input
                net_output = self.forward(net_input)

                del net_input
                loss = loss_fn(net_output, labels)
                metric = metric_fn(net_output, labels)
                accuracies = accuracies_fn(net_output, labels)
                del net_output
                running_loss += loss.item()
                running_metric += metric.item()
                running_acc += accuracies.detach().cpu().numpy()
                del loss
                del metric

        return running_loss / len(val_loader), running_metric / len(val_loader), running_acc / len(val_loader)

    @staticmethod
    def predict_batch(net, test_loader, DEVICE):
        net.eval()
        conc_output = []
        conc_ID = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting test set..."):
                net_input = [b.to(DEVICE) for b in batch[0]]
                conc_ID.extend(list(batch[0].detach().cpu().numpy()))
                # evaluate the network over the input
                conc_output.extend(list(net(net_input).detach().cpu().numpy()))

        return conc_ID, conc_output


class BasePlainNet3D(BaseNetwork):
    def __init__(self, use_apex):
        super().__init__(use_apex=use_apex)
        self.net_3d = None  # Define plain network here

    def forward(self, inputs, mask=None):
        inputs = inputs[0]  # Extract brain from list of one
        return self.net_3d(inputs)


class PlainNetSiamese(BaseNetwork):
    def __init__(self, use_apex=False):
        super().__init__(use_apex)
        self.net_3d = None
        self.regressor = None

    def forward(self, inputs, mask=None):
        # Extract brains
        brains = inputs[0]
        # Go with siamese network, one for each brain's color channel
        brains = torch.cat([F.relu(self.net_3d(brain.unsqueeze(1)), inplace=True) for brain in brains.transpose(1, 0)], dim=1)
        # ALREADY TRIED THIS APPROACH, TESTED OK (BUT NOT FOR BATCHNORM). NO PERFORMANCE INCREASING FOUND.
        # brains = brains.view(brains.size(0)*brains.size(1), brains.size(2), brains.size(3), brains.size(4)).unsqueeze(1)
        # brains = F.relu(self.net_3d(brains))
        # brains = brains.view(inputs[0].size(0), -1)
        brains = torch.clamp(self.regressor(brains), -0.1, 5)
        return brains.exp()

