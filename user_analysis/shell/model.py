#coding=utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.autograd import Variable
from ipdb import set_trace
from torch.nn import utils as nn_utils


class LSTMBuyRedeem(nn.Module):

    def __init__(self, nav_dim, hidden_dim, tag_size):
        super(LSTMBuyRedeem, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes nav feature as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(nav_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).cuda(),
                torch.zeros(1, 1, self.hidden_dim).cuda())


    def forward(self, nav_ser):
        lstm_out, self.hidden = self.lstm(
            nav_ser.view(len(nav_ser), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(nav_ser), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
