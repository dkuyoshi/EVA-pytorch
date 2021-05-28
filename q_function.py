import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl import action_value
from pfrl.nn.mlp import MLP
from pfrl.q_function import StateQFunction
from pfrl.initializers import init_chainer_default
from value_buffer import ValueBuffer

def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias


class CNN(nn.Module):
    def __init__(self, n_input_channels=4, activation=F.relu, n_hidden=256, bias=0.1, ):
        super(CNN, self).__init__()
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_hidden = n_hidden

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 16, 8, stride=4),
                nn.Conv2d(16, 32, 4, stride=2),
                nn.Conv2d(32, 32, 3, stride=1),
            ]
        )

        self.head = nn.Linear(1568, 256)

        self.conv_layers.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)

        h = self.head(h)

        return h


class QNet(nn.Module):
    def __init__(self, n_input_channels, n_actions, n_hidden=256):
        super().__init__()
        self.h_out = CNN(n_input_channels, activation=F.relu, n_hidden=n_hidden)
        self.q_out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        self.embedding = self.h_out(x)
        q = self.q_out(F.relu(self.embedding))
        return action_value.DiscreteActionValue(q)


class QFunction(nn.Module):
    def __init__(self, n_input_channels, n_actions, device, LRU=False, n_hidden=256, lambdas=0.5, capacity=2000, num_neighbors=5):
        super().__init__()
        self.q_func = QNet(n_input_channels, n_actions, n_hidden)
        # Call the initialization of the value buffer that outputs the non-parametric Q value
        self.non_q = ValueBuffer(capacity, num_neighbors, n_hidden, n_actions, device, LRU=LRU)

        self.lambdas = lambdas
        self.n_actions = n_actions
        self.capacity = capacity

    def forward(self, x, eva=False):
        q = self.q_func(x)
        self.embedding = self.get_embedding()
        if not eva or self.lambdas == 0 or self.lambdas == 1:
            return q

        qnp = self.non_q.get_q(self.embedding.detach().cpu().numpy())
        # Q-value adjustment
        qout = self.lambdas * q.q_values + (1 - self.lambdas) * qnp
        return action_value.DiscreteActionValue(qout)

    def get_embedding(self):
        return self.q_func.embedding
        
