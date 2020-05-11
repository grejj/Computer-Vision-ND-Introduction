#!/usr/bin/env python3

import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.autograd import Variable

# for random variables
torch.manual_seed(2)

# define an LSTM with input dim of 4 and hidden dim of 3 (output 3 values)
input_dim = 4
hidden_dim = 3
lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)

# make 5 input sequences of 4 random values each
inputs_list = [torch.randn(1, 4) for _ in range(5)]
print("inputs: {}\n".format(inputs_list))

# initialize hidden state (1 layer, 1 batch size, 3 outputs)
# first tensor is hidden state (h0) and second tensor initializes cell memory (c0)
h0 = torch.randn(1, 1, hidden_dim)
c0 = torch.randn(1, 1, hidden_dim)

h0 = Variable(h0)
c0 = Variable(c0)
# step through sequence
for i in inputs_list:
    
    i = Variable(i)

    out, hidden = lstm(i.view(1, 1, -1), (h0, c0))
    print("out: {}\n".format(out))
    print("hidden: {}\n".format(hidden))

# process all inputs at once instead as for loop is not very efficient

# turn inputs into tensor with 5 rows of data and 2nd dimension 1 for batch size
inputs = torch.cat(inputs_list).view(len(inputs_list), 1, -1)

print("inputs size: {}".format(inputs.size()))
print("inputs: {}".format(inputs))

# initialize hidden state
h0 = torch.randn(1, 1, hidden_dim)
c0 = torch.randn(1, 1, hidden_dim)

# wrap everything
inputs = Variable(inputs)
h0 = Variable(h0)
c0 = Variable(c0)

# get outputs and hidden state
out, hidden = lstm(inputs, (h0, c0))

print("out: {}".format(out))
print("hidden: {}".format(hidden))
