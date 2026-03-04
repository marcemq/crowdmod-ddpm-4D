import torch
from torch import nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        ConvLSTM cell.

        Args:
            input_size (int, int): Height and width of input tensor as (height, width).
            input_dim int: Number of channels of input tensor.
            hidden_dim int: Number of channels of hidden state.
            kernel_size (tuple): convolution kernel size
            bias (bool): whether to use bias
        """
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.padding = 1, 1
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Single convolution computing all 4 gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state and cell state.
        Returns:
            (h, c)
        """
        h = torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device)
        return (h, c)

    def forward(self, input, hidden_state):
        """
        Args:
            input: (B, C_in, H, W)
            hidden_state: tuple (h_prev, c_prev)

        Returns:
            (h_next, c_next)
        """

        h_prev, c_prev = hidden_state

        # Concatenate along channel axis
        combined = torch.cat([input, h_prev], dim=1)

        conv_output = self.conv(combined)

        # Split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        input_gate  = self.sigmoid(cc_i)
        forget_gate = self.sigmoid(cc_f)
        output_gate = self.sigmoid(cc_o)
        candidate   = self.tanh(cc_g)

        # Cell update
        c_next = forget_gate * c_prev + input_gate * candidate

        # Hidden update
        h_next = output_gate * self.tanh(c_next)

        return (h_next, c_next)