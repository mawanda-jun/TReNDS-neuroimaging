import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from DenseNet3D import DenseNet3D
from ResNet import resnet3d_10
from tqdm import tqdm


class BaseNetwork(nn.Module):
    def __init__(self):
        # inizializzazione classe base - si fa sempre
        super(BaseNetwork, self).__init__()

    def forward(self, inputs, mask=None):
        pass

    @staticmethod
    def get_input(batch, DEVICE):
        pass

    def train_batch(self, net, train_loader, loss_fn, metric_fn, optimizer, DEVICE) -> (torch.Tensor, torch.Tensor):
        """
        Define training method only once. The only method that must be done is how the training gets the training inputs
        :param net:
        :param train_loader:
        :param loss_fn:
        :param metric_fn:
        :param optimizer:
        :param DEVICE:
        :return:
        """
        net.to(DEVICE)
        net.train()
        conc_losses = []
        conc_metrics = []
        for batch in tqdm(train_loader, desc='Training...'):
            net_input = self.get_input(batch, DEVICE)

            labels = batch['label'].to(DEVICE)

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
            conc_metrics.append(metric.item())

            del loss
            del metric

        return torch.mean(torch.tensor(conc_losses)), torch.mean(torch.tensor(conc_metrics))

    def val_batch(self, net, val_loader, loss_fn, metric_fn, DEVICE) -> (torch.Tensor, torch.Tensor):
        net.to(DEVICE)
        net.eval()
        conc_losses = []
        conc_metrics = []

        for batch in tqdm(val_loader, desc='Validating...'):
            net_input = self.get_input(batch, DEVICE)
            labels = batch['label'].to(DEVICE)

            # evaluate the network over the input
            net_output = net(net_input)
            del net_input
            loss = loss_fn(net_output, labels)
            metric = metric_fn(net_output, labels)
            del net_output
            conc_losses.append(loss.item())
            conc_metrics.append(metric.item())
            del loss
            del metric

        return torch.mean(torch.tensor(conc_losses)), torch.mean(torch.tensor(conc_metrics))

    def predict_batch(self, net, test_loader, DEVICE) -> (np.ndarray, np.ndarray):
        net.eval()
        conc_output = []
        conc_ID = []

        for batch in test_loader:
            net_input = self.get_input(batch, DEVICE)
            conc_ID.extend(list(batch['ID'].detach().cpu().numpy()))
            # evaluate the network over the input
            conc_output.extend(list(net(net_input).detach().cpu().numpy()))

        return conc_ID, conc_output


class ShallowNet(BaseNetwork):
    def __init__(self,
                 dropout_prob=0.,
                 ):
        # inizializzazione classe base - si fa sempre
        super(ShallowNet, self).__init__()

        # definiamo i layer della rete
        self.FC1 = nn.Linear(in_features=1378 + 26, out_features=2048)
        self.FC2 = nn.Linear(in_features=2048, out_features=512)
        # self.FC3 = nn.Linear(in_features=512, out_features=128)
        # self.FC4 = nn.Linear(in_features=1024, out_features=128)
        self.drop1 = nn.Dropout(p=0)
        self.drop2 = nn.Dropout(p=dropout_prob)
        # self.drop3 = nn.Dropout(p=dropout_prob)
        # self.drop4 = nn.Dropout(p=dropout_prob)
        self.regressor = nn.Linear(in_features=512, out_features=5)

    def forward(self, inputs, mask=None):
        fnc = inputs['fnc']
        sbm = inputs['sbm']
        # strato 1: FC+dropout]ReLu
        x = self.FC1(torch.cat([fnc, sbm], dim=1))
        x = self.drop1(x)
        x = F.relu(x)
        # strato 2: FC+dropout+ReLu
        x = self.FC2(x)
        x = self.drop2(x)
        x = F.relu(x)
        # strato 3: FC+dropout + ReLu
        # x = self.FC3(x)
        # x = self.drop3(x)
        # x = F.relu(x)
        # strato 4: FC+dropout + ReLu
        # x = self.FC4(x)
        # x = self.drop4(x)
        # x = F.relu(x)
        # strato 4: regressore
        x = self.regressor(x)

        return F.relu(x)

    @staticmethod
    def get_input(batch, DEVICE):
        return {'fnc': batch['fnc'].to(DEVICE),
                'sbm': batch['sbm'].to(DEVICE)
                }


class CustomDenseNet3D(BaseNetwork):
    def __init__(self, dropout_prob=0., *args, **kwargs):
        # inizializzazione classe base - si fa sempre
        super(CustomDenseNet3D, self).__init__()
        # The in-channel was 2 and out features 32, so a growth of 16. It's maybe too heavy for my computer,
        # So I apply a growth factor of 2 in the first layer
        # out_net and fc_dim can be inherited by other networks
        self.net_3d = DenseNet3D(num_init_features=16, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2)
        self.fc_dim = 26 + 122
        # definiamo i layer della rete
        self.FC1 = nn.Linear(in_features=self.fc_dim, out_features=512)
        self.FC2 = nn.Linear(in_features=512, out_features=128)
        # self.FC3 = nn.Linear(in_features=512, out_features=128)
        # self.FC4 = nn.Linear(in_features=1024, out_features=128)
        self.drop1 = nn.Dropout(p=0.)
        self.drop2 = nn.Dropout(p=dropout_prob)
        # self.drop3 = nn.Dropout(p=dropout_prob)
        # self.drop4 = nn.Dropout(p=dropout_prob)
        self.regressor = nn.Linear(in_features=128, out_features=5)

    def forward(self, inputs, mask=None):
        sbm = inputs['sbm']
        brain = inputs['brain']
        x_brain = self.net_3d(brain)
        # strato 1: FC+dropout]ReLu
        x = self.FC1(torch.cat([sbm, x_brain.view(x_brain.shape[0], -1)], dim=1))
        x = self.drop1(x)
        x = F.relu(x)
        # strato 2: FC+dropout+ReLu
        x = self.FC2(x)
        x = self.drop2(x)
        x = F.relu(x)
        # strato 3: FC+dropout + ReLu
        # x = self.FC3(x)
        # x = self.drop3(x)
        # x = F.relu(x)
        # strato 4: FC+dropout + ReLu
        # x = self.FC4(x)
        # x = self.drop4(x)
        # x = F.relu(x)
        # strato 4: regressore
        x = self.regressor(x)

        return F.relu(x)

    @staticmethod
    def get_input(batch, DEVICE):
        return {
            # 'fnc': batch['fnc'].to(DEVICE),
            'sbm': batch['sbm'].to(DEVICE),
            'brain': batch['brain'].to(DEVICE)
        }


class CustomResNet3D(CustomDenseNet3D):
    def __init__(self, dropout_prob=0.):
        super(CustomDenseNet3D, self).__init__(dropout_prob)
        self.net_3d = resnet3d_10()
        self.fc_dim = 26 + 0
