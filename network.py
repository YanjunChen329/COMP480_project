import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(1, 100).double()
        self.fc2 = nn.Linear(100, 100).double()
        self.fc3 = nn.Linear(100, 100).double()
        self.fc4 = nn.Linear(100, 2).double()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sin(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


class ModLayer(nn.Module):
    def __init__(self, init_K, N=1000, bias=0.5):
        super(ModLayer, self).__init__()
        self.K = Parameter(torch.tensor(init_K, dtype=torch.float))
        # self.K = torch.tensor(init_K, dtype=torch.float)
        # print(self.K)
        self.range_array = torch.arange(1, N + 1).type(torch.FloatTensor)
        self.range_diag = torch.diag(1./self.range_array)
        # print(self.range_diag)
        self.bias = bias

    def forward(self, x):
        # self.K = Parameter(torch.tensor(round(self.K.item()), dtype=torch.float))
        x += self.bias
        sin_array = (2 * math.pi / self.K) * self.range_array.view(1, -1)
        x = torch.mm(x, sin_array)
        x = torch.sin(x)
        x = torch.mm(x, self.range_diag)
        x = torch.sum(x, dim=1)
        x *= -self.K/math.pi
        x += self.K/2. - self.bias
        return x


class HashNet(nn.Module):
    def __init__(self, guess_K, guess_a, guess_b):
        super(HashNet, self).__init__()
        self.a = Parameter(torch.tensor(guess_a, dtype=torch.float))
        self.b = Parameter(torch.tensor(guess_b, dtype=torch.float))
        self.Modlayer = ModLayer(guess_K)

    def forward(self, x):
        x = self.a*x + self.b
        x = self.Modlayer(x)
        return x


class BF_Net(nn.Module):
    def __init__(self, N=10000, hn=100, use_cuda=True):
        super(BF_Net, self).__init__()
        self.fc1 = nn.Linear(1, hn).double()
        self.fc2 = nn.Linear(hn, hn).double()
        self.fc3 = nn.Linear(hn, hn).double()
        self.fc4 = nn.Linear(hn, 2).double()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

        range_array = torch.arange(1, N + 1).type(torch.DoubleTensor)
        self.mod_mtx = (2*math.pi*range_array/N).view(hn, hn)
        # self.mod_mtx = arr
        # for _ in range(hn - 1):
        #     self.mod_mtx = torch.cat((self.mod_mtx, arr), dim=0)

        if use_cuda:
            self.mod_mtx = self.mod_mtx.cuda()
        # print(self.mod_mtx)
        # print(self.mod_mtx.shape)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = torch.mm(x, self.mod_mtx)
        x = torch.sin(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x