import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from apex import amp


class BaseNetwork(nn.Module):
    """
    Base class to define common methods among all networks
    """
    def __init__(self, use_apex=False):
        # inizializzazione classe base - si fa sempre
        super().__init__()
        self.use_apex = use_apex

    def forward(self, inputs, mask=None):
        pass

    def train_batch(self, net, train_loader, loss_fn, metric_fn, accuracies_fn, optimizer, scheduler, DEVICE) -> (torch.Tensor, torch.Tensor):
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
        net.to(DEVICE)
        net.train()
        running_loss = 0
        running_metric = 0
        running_acc = np.zeros(5)
        for batch in tqdm(train_loader, desc='Training...'):
            net_input = [b.to(DEVICE) for b in batch[0]]

            labels = batch[1].to(DEVICE)

            # forward pass
            net_output = net(net_input)

            del net_input

            # update networks
            loss = loss_fn(net_output, labels)
            metric = metric_fn(net_output[0], labels)
            accuracies = accuracies_fn(net_output[0], labels)

            del net_output

            # clear previous recorded gradients
            optimizer.zero_grad()

            # backward pass
            if self.use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # update optimizer
            optimizer.step()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)

            running_loss += loss.item()
            running_metric += metric.item()
            running_acc += accuracies.detach().cpu().numpy()

            del loss
            del metric

            # Update scheduler
            scheduler.step()

        return running_loss / len(train_loader), running_metric / len(train_loader), running_acc / len(train_loader)

    @staticmethod
    def val_batch(net, val_loader, loss_fn, metric_fn, accuracies_fn, DEVICE) -> (torch.Tensor, torch.Tensor):
        net.to(DEVICE)
        net.eval()
        running_loss = 0
        running_metric = 0
        running_acc = np.zeros(5)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating...'):
                net_input = [b.to(DEVICE) for b in batch[0]]
                labels = batch[1].to(DEVICE)

                # evaluate the network over the input
                net_output = net(net_input)
                del net_input
                loss = loss_fn(net_output, labels)
                metric = metric_fn(net_output[0], labels)
                accuracies = accuracies_fn(net_output[0], labels)
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
    def __init__(self, use_apex=False):
        super().__init__(use_apex=use_apex)
        self.net_3d = None  # Define plain network here

    def forward(self, inputs, mask=None):
        _, brain = inputs  # take only brain
        return self.net_3d(brain)

