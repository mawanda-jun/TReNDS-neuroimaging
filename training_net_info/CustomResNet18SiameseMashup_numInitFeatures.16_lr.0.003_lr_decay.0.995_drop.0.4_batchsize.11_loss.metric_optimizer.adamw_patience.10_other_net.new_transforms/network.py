import torch
from torch import nn
import torch.nn.functional as F
from _collections import OrderedDict

from base_networks import BaseNetwork, BaseCustomNet3D, BasePlainNet3D
from ResNet import resnet50, resnet3d_10, resnet101, resnet34, resnet18
from DenseNet3D import densenet121, densenet_custom, densenet161


class ShallowNet(BaseNetwork):
    def __init__(self,
                 dropout_prob=0.,
                 ):
        # inizializzazione classe base - si fa sempre
        super().__init__()

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
        sbm = inputs[1]
        fnc = inputs[-2]
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
        # return sbm, fnc
        return batch[1].to(DEVICE), batch[2].to(DEVICE)


class CustomDenseNet3D(BaseCustomNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64, *args, **kwargs):
        # inizializzazione classe base - si fa sempre
        super().__init__()
        # The in-channel was 2 and out features 32, so a growth of 16. It's maybe too heavy for my computer,
        # So I apply a growth factor of 2 in the first layer
        # out_net and fc_dim can be inherited by other networks
        self.net_3d = densenet121(num_init_features=num_init_features, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2)
        self.__net_3d_out_dim = 128
        # I decided to concatenate the logits from net_3d with the logits of the resulting fc layer which has already
        # processed the sbm. Therefore, I will concatenate it on the regressor directly
        # definiamo i layer della rete
        self.FC1 = nn.Linear(in_features=26 + self.__net_3d_out_dim, out_features=2048)
        self.FC2 = nn.Linear(in_features=2048, out_features=128)
        # self.FC3 = nn.Linear(in_features=512, out_features=128)
        # self.FC4 = nn.Linear(in_features=1024, out_features=128)
        self.drop1 = nn.Dropout(p=0.)
        self.drop2 = nn.Dropout(p=dropout_prob)
        # self.drop3 = nn.Dropout(p=dropout_prob)
        # self.drop4 = nn.Dropout(p=dropout_prob)
        self.regressor = nn.Linear(in_features=128, out_features=5, bias=True)

    @property
    def net_3d_out_dim(self):
        return self.net_3d_out_dim

    @net_3d_out_dim.setter
    def net_3d_out_dim(self, value):
        """
        We need to update the regressor dimension in case we want to change the dimension of the 3D network result
        :param value:
        :return:
        """
        self.__net_3d_out_dim = value
        self.FC1 = nn.Linear(in_features=26 + self.__net_3d_out_dim, out_features=2048)

    def forward(self, inputs, mask=None):
        sbm = inputs['sbm']
        brain = inputs['brain']
        x_brain = F.relu(self.net_3d(brain))
        # strato 1: FC+dropout]ReLu
        x = self.FC1(torch.cat([x_brain.view(x_brain.shape[0], -1), sbm], dim=1))
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


class CustomResNet3D10(BaseCustomNet3D):
    """
    In this network I try to understand age from the sbm alone, and let the network extract the other information
    directly from the brain images.
    """
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = resnet3d_10(dropout_prob=dropout_prob, num_init_features=num_init_features, num_class=4)
        self.sbm_features = nn.Sequential(OrderedDict([
            ('FC1', nn.Linear(26, 256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('FC2', nn.Linear(256, 1))
        ]))

    def forward(self, inputs, mask=None):
        sbm, brain = inputs
        x_brain = F.relu(self.net_3d(brain))
        x_sbm = F.relu(self.sbm_features(sbm))
        return torch.cat([x_sbm, x_brain], dim=1)


class CustomResNet18Siamese(BaseCustomNet3D):
    """
    In this network I try to understand age from the sbm alone, and let the network extract the other information
    directly from the brain images.
    """
    def __init__(self, dropout_prob=0., num_init_features=16):
        super().__init__()
        num_brain_features = 8
        self.dropout_prob = dropout_prob
        self.net_3d = resnet18(dropout_prob=dropout_prob, in_channels=1, num_init_features=num_init_features, num_class=num_brain_features)
        self.sbm_fnc_features = nn.Sequential(OrderedDict([
            ('FC1', nn.Linear(26+1378, 2048, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('FC2', nn.Linear(2048, 512, bias=False))
        ]))
        self.regressor = nn.Linear(num_brain_features*53 + 512, 5, bias=True)

    def forward(self, inputs, mask=None):
        # Extract sbm, brains and fnc
        sbm, brains, fnc = inputs
        # Go with siamese network, one for each brain's color channel
        x_brain = []
        for brain in brains.transpose(1, 0):
            x_brain.append(F.relu(self.net_3d(brain.unsqueeze(1))))
        x_brain = torch.cat(x_brain, dim=1)
        # Calculate features for fnc and sbm features
        x_sbm_fnc = F.relu(self.sbm_fnc_features(torch.cat([sbm, fnc], dim=1)))
        # Do dropout on both the outputs
        x_brain = F.dropout(x_brain, self.dropout_prob, self.training)
        x_sbm_fnc = F.dropout(x_sbm_fnc, self.dropout_prob, self.training)
        # Concat everything and feed the regressor
        x = F.relu(self.regressor(torch.cat([x_sbm_fnc, x_brain], dim=1)))
        return x


class CustomResNet18SiameseMashup(BaseCustomNet3D):
    """
    In this network I try to understand age from the sbm alone, and let the network extract the other information
    directly from the brain images.
    """
    def __init__(self, dropout_prob=0., num_init_features=16):
        super().__init__()
        num_brain_features = 8
        self.dropout_prob = dropout_prob
        self.net_3d = resnet18(dropout_prob=dropout_prob, in_channels=1, num_init_features=num_init_features, num_class=num_brain_features)
        self.sbm_fnc_features = nn.Linear(26+1378, 2048, bias=False)
        self.FC_mashup = nn.Linear(2048 + num_brain_features*53, 512, bias=True)
        self.regressor = nn.Linear(512, 5, bias=True)

    def forward(self, inputs, mask=None):
        # Extract sbm, brains and fnc
        sbm, brains, fnc = inputs
        # Go with siamese network, one for each brain's color channel
        x_brain = []
        for brain in brains.transpose(1, 0):
            x_brain.append(F.relu(self.net_3d(brain.unsqueeze(1))))
        x_brain = torch.cat(x_brain, dim=1)
        # Calculate features for fnc and sbm features
        x_sbm_fnc = F.relu(self.sbm_fnc_features(torch.cat([sbm, fnc], dim=1)))
        # Do dropout on both the outputs
        x_brain = F.dropout(x_brain, self.dropout_prob, self.training)
        x_sbm_fnc = F.dropout(x_sbm_fnc, self.dropout_prob, self.training)
        # Concat everything and feed the mashup
        x = F.relu(self.FC_mashup(torch.cat([x_sbm_fnc, x_brain], dim=1)))
        # Apply dropout
        x = F.dropout(x, self.dropout_prob, self.training)
        x = F.relu(self.regressor(x))
        return x


class CustomResNet3D50(CustomDenseNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__(dropout_prob)
        self.net_3d = resnet50(dropout_prob=dropout_prob, num_init_features=num_init_features)
        self.net_3d_out_dim = 2048


class PlainDenseNet3DCustom(BasePlainNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = densenet_custom(
            dropout_prob=dropout_prob,
            num_init_features=num_init_features,
            num_classes=5,
        )


class PlainDenseNet3D121(BasePlainNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = densenet121(
            dropout_prob=dropout_prob,
            num_init_features=num_init_features,
            num_classes=5,
        )


class PlainDenseNet3D161(BasePlainNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = densenet161(
            dropout_prob=dropout_prob,
            num_init_features=num_init_features,
            num_classes=5,
        )


class PlainResNet3D10(BasePlainNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = resnet3d_10(num_class=5, dropout_prob=dropout_prob, num_init_features=num_init_features)


class PlainResNet3D34(BasePlainNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = resnet34(num_class=5, dropout_prob=dropout_prob, num_init_features=num_init_features)


class PlainResNet3D50(BasePlainNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = resnet50(num_class=5, dropout_prob=dropout_prob, num_init_features=num_init_features)


class PlainResNet3D101(BasePlainNet3D):
    def __init__(self, dropout_prob=0., num_init_features=64):
        super().__init__()
        self.net_3d = resnet101(num_class=5, dropout_prob=dropout_prob, num_init_features=num_init_features)
