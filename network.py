import torch
from torch import nn
import torch.nn.functional as F


class ShallowNet(nn.Module):
    def __init__(self,
                 fnc_dim,
                 sbm_dim,
                 dropout_prob=0.,
                 ):
        # inizializzazione classe base - si fa sempre
        super(ShallowNet, self).__init__()

        # definiamo i layer della rete
        self.FC1 = nn.Linear(in_features=fnc_dim+sbm_dim, out_features=2048)
        self.FC2 = nn.Linear(in_features=2048, out_features=1048)
        self.drop = nn.Dropout(p=dropout_prob)
        self.regressor = nn.Linear(in_features=1048, out_features=5)

    def forward(self, inputs, mask=None):
        fnc = inputs['fnc']
        sbm = inputs['sbm']
        # strato 1: FC+ReLu
        x = self.FC1(torch.cat([fnc, sbm], dim=1))
        x = F.relu(x)
        # strato 2: FC+dropout+ReLu
        x = self.FC2(x)
        x = self.drop(x)
        x = F.relu(x)
        # strato 3: regressore
        x = self.regressor(x)

        return x


def train_batch(net, train_loader, loss_fn, metric_fn, optimizer, DEVICE):
    net.train()
    for batch in train_loader:
        net_input = {'fnc': batch['fnc'].to(DEVICE),
                     'sbm': batch['sbm'].to(DEVICE)
                     }
        labels = batch['label'].to(DEVICE)

        # forward pass
        net_output = net(net_input)

        # update networks
        loss = loss_fn(net_output, labels)
        metric = metric_fn(net_output, labels)

        # clear previous recorded gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update optimizer
        optimizer.step()

        return loss, metric


def val_batch(net, val_loader, loss_fn, metric_fn, DEVICE):
    net.eval()
    conc_output = []
    conc_label = []

    for batch in val_loader:
        net_input = {'fnc': batch['fnc'].to(DEVICE),
                     'sbm': batch['sbm'].to(DEVICE)
                     }
        labels = batch['label'].to(DEVICE)

        # evaluate the network over the input
        net_output = net(net_input)

        conc_output.append(net_output)
        conc_label.append(labels)

    conc_output = torch.cat(conc_output)
    conc_label = torch.cat(conc_label)

    loss = loss_fn(conc_output, conc_label)
    metric = metric_fn(conc_output, conc_label)

    return loss, metric









