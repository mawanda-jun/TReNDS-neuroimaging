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
        x = self.FC1(torch.cat([fnc, sbm]))
        x = F.relu(x)
        # strato 2: FC+dropout+ReLu
        x = self.FC2(x)
        x = self.drop(x)
        x = F.relu(x)
        # strato 3: regressore
        x = self.regressor(x)

        return x