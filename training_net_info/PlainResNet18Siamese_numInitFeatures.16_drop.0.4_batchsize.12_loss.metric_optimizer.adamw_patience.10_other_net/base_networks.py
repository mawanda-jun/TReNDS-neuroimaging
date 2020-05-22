import torch.nn as nn
import torch
from tqdm import tqdm


class BaseNetwork(nn.Module):
    """
    Base class to define common methods among all networks
    """
    def __init__(self):
        # inizializzazione classe base - si fa sempre
        super().__init__()

    def forward(self, inputs, mask=None):
        pass

    @staticmethod
    def get_input(batch, DEVICE):
        pass

    def train_batch(self, net, train_loader, loss_fn, metric_fn, optimizer, scheduler, DEVICE) -> (torch.Tensor, torch.Tensor):
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
        conc_losses = []
        conc_metrics = []
        for batch in tqdm(train_loader, desc='Training...'):
            net_input = self.get_input(batch, DEVICE)

            labels = batch[-1].to(DEVICE)

            # forward pass
            net_output = net(net_input)

            del net_input

            # update networks
            loss = loss_fn(net_output, labels)
            metric = metric_fn(net_output, labels)

            del net_output

            # clear previous recorded gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update optimizer
            optimizer.step()

            conc_losses.append(loss.item())
            # conc_metrics.append(metric.item())
            conc_metrics.append(metric)

            del loss
            # del metric

            # Update scheduler
            scheduler.step()

        stacked_metrics = torch.stack(conc_metrics, dim=0).detach().cpu()
        del conc_metrics

        return torch.mean(torch.tensor(conc_losses)), torch.mean(stacked_metrics, dim=0).numpy()

    def val_batch(self, net, val_loader, loss_fn, metric_fn, DEVICE) -> (torch.Tensor, torch.Tensor):
        net.to(DEVICE)
        net.eval()
        conc_losses = []
        conc_metrics = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating...'):
                net_input = self.get_input(batch, DEVICE)
                labels = batch[-1].to(DEVICE)

                # evaluate the network over the input
                net_output = net(net_input)
                del net_input
                loss = loss_fn(net_output, labels)
                metric = metric_fn(net_output, labels)
                del net_output
                conc_losses.append(loss.item())
                # conc_metrics.append(metric.item())
                conc_metrics.append(metric)
                del loss
                # del metric

            stacked_metrics = torch.stack(conc_metrics, dim=0).detach().cpu()
            del conc_metrics

        return torch.mean(torch.tensor(conc_losses)), torch.mean(stacked_metrics, dim=0).numpy()

    def predict_batch(self, net, test_loader, DEVICE):
        net.eval()
        conc_output = []
        conc_ID = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting test set..."):
                net_input = self.get_input(batch, DEVICE)
                conc_ID.extend(list(batch[0].detach().cpu().numpy()))
                # evaluate the network over the input
                conc_output.extend(list(net(net_input).detach().cpu().numpy()))

        return conc_ID, conc_output


class BaseCustomNet3D(BaseNetwork):
    """
    Base class to define how input is taken for 3D networks
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_input(batch, DEVICE):
        # return sbm, brain, fnc
        return batch[1].to(DEVICE), batch[2].to(DEVICE), batch[3].to(DEVICE)


class BasePlainNet3D(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.net_3d = None  # Define plain network here

    @staticmethod
    def get_input(batch, DEVICE):
        # return brain
        return batch[2].to(DEVICE)

    def forward(self, inputs, mask=None):
        return self.net_3d(inputs)

