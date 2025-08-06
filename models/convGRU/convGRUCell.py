
import os
import torch
from torch import nn
from torch.autograd import Variable

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvGRU cell.

        Args:
            input_size (int, int): Height and width of input tensor as (height, width).
            input_dim int: Number of channels of input tensor.
            hidden_dim int: Number of channels of hidden state.
            kernel_size (int, int): Size of the convolutional kernel.
            bias bool: Whether or not to add the bias.
            dtype torch.cuda.FloatTensor or torch.FloatTensor: Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = 1, 1
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.reset_gate = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=self.hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.update_gate = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                     out_channels=self.hidden_dim,
                                     kernel_size=kernel_size,
                                     padding=self.padding,
                                     bias=self.bias)
        
        self.conv_cand = nn.Conv2d(in_channels=input_dim+hidden_dim,
                                   out_channels=self.hidden_dim, # for candidate neural memory
                                   kernel_size=kernel_size,
                                   padding=self.padding,
                                   bias=self.bias)
     
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, self.height, self.width)

    def forward(self, input, hidden_state):
        """
        Forward function.

        Args:
            input (b, c, h, w): input is actually the target_model
            hidden_state (b, c_hidden, h, w): current hidden and cell states respectively

        Returns:
            - new_hidden_state: next hidden state
        """
        combined = torch.cat([input, hidden_state], dim=1)

        reset_gate = self.sigmoid(self.reset_gate(combined))
        update_gate = self.sigmoid(self.update_gate(combined))

        combined_reset = torch.cat([input, reset_gate*hidden_state], dim=1)
        candidate = self.tanh(self.conv_cand(combined_reset))

        new_hidden_state = (1 - update_gate) * candidate + update_gate * hidden_state

        return new_hidden_state
