import torch.nn as nn
import torch
from torchmetrics import Accuracy
# from torchvision.datasets import FashionMNIST
# from torchvision import transforms
# from torch.utils.data import  DataLoader
from torch.functional import F
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 15
import sys
from sklearn.utils import shuffle

cuda = True if torch.cuda.is_available() else False
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.manual_seed(0)


class DataModule:
    def __init__(self,dim, normalize=True, **kwargs):
        # a 3-D X : (batchs, numbers, 1)
        # and we will follow the -π to π for initial experiments
        if kwargs['f'] == 'delta':
            self.p = self.delta()
        elif kwargs['f'] == 'gauss':
            self.X = np.random.normal(loc=kwargs['mu'], scale=kwargs['sigma'], size=dim)
            if normalize:
                self.scale = (self.X.max() - self.X.min())
                self.X = self.X / self.scale
            self.p = self.gauss(kwargs['mu'], kwargs['sigma'])
        elif kwargs['f'] == 'uniform':
            self.X = np.random.uniform(kwargs['min'], kwargs['max'], size=dim)
            if normalize:
                self.scale = (self.X.max() - self.X.min())
                self.X = self.X / self.scale
            self.p = self.uniform(kwargs['min'], kwargs['max'])
        train_ratio = int(0.8*len(self.X))
        self.X_train, self.X_test = self.X[:train_ratio], self.X[train_ratio:]
        self.X_train, self.X_test  = Variable(Tensor(self.X_train)),Variable(Tensor(self.X_test))

    def gauss(self, mu, sigma):
        return np.exp(-0.5*((self.X-mu)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
                nn.Linear(1, 4),
                nn.LeakyReLU(0.01),
                nn.Linear(4, 4),
                nn.Dropout(0.5),
                nn.Tanh(),
                nn.Linear(4,4),
                nn.Dropout(0.5),
                nn.Tanh(),
                nn.Linear(4, 1),
                nn.Tanh()
            )

    def forward(self, z):
        y_gen = self.model(z)
        return y_gen


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1, 4),   #8,
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, fn):
        validity = self.model(fn)
        return validity


adversarial_loss = nn.BCELoss()
