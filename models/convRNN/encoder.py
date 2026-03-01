import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels, enc_kernels, num_layers, device, cell_class, bias=True):
        """
        Args:
        input_size (int, int): Height and width of input tensor as (height, width).
        input_dim int: Number of channels of input tensor.
        hidden_dim int: Number of channels of hidden state.
        kernel_size (int, int): Size of the convolutional kernel.
        num_layers int: Number of ConvLSTM layers
        bias bool: Whether or not to add the bias.
        """
        super(Encoder, self).__init__()

        # Make sure that both `enc_kernels` and `hidden_dim` are lists having len == num_layers
        if not len(enc_kernels) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.enc_kernels = enc_kernels
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.cell_class = cell_class
        self.leakyReLU =nn.LeakyReLU(0.02)

        encoder_cell_list = []
        two_stride = 2

        # Encoder layers
        # Conv 1
        encoder_cell_list.append(
            nn.Conv2d(in_channels=self.input_channels,
                      out_channels=self.hidden_channels[0],
                      kernel_size=self.enc_kernels[0],
                      padding=1,
                      bias=self.bias))
        # RNN 1
        encoder_cell_list.append(
            self.cell_class(input_size=(self.height, self.width),
                            input_dim=self.hidden_channels[0],
                            hidden_dim=self.hidden_channels[1],
                            kernel_size=self.enc_kernels[1],
                            bias=self.bias))
        # Down 1
        encoder_cell_list.append(
            nn.Conv2d(in_channels=self.hidden_channels[1],
                              out_channels=self.hidden_channels[2],
                              kernel_size=self.enc_kernels[2],
                              padding=1,
                              stride=two_stride,
                              bias=self.bias))
        # RNN 2
        encoder_cell_list.append(
            self.cell_class(input_size=(self.height//two_stride, self.width//two_stride),
                            input_dim=self.hidden_channels[1],
                            hidden_dim=self.hidden_channels[3],
                            kernel_size=self.enc_kernels[3],
                            bias=self.bias))
        # Down 2
        encoder_cell_list.append(
            nn.Conv2d(in_channels=self.hidden_channels[3],
                      out_channels=self.hidden_channels[4],
                      kernel_size=self.enc_kernels[4],
                      padding=1,
                      stride=two_stride,
                      bias=self.bias))
        # RNN 3
        encoder_cell_list.append(
            self.cell_class(input_size=(self.height//(two_stride*2), self.width//(two_stride*2)),
                            input_dim=self.hidden_channels[3],
                            hidden_dim=self.hidden_channels[5],
                            kernel_size=self.enc_kernels[5],
                            bias=self.bias))
        
        self.encoder_cell_list = nn.ModuleList(encoder_cell_list)

    def forward(self, input, hidden_state):
        """
        :param input_tensor (b, h, w, c, t) 
        :param hidden_state:
        :return: last_state_list
        """
        last_state_list   = []
        obs_seq_len = input.size(4)

        for t in range(obs_seq_len):
            # -----------------
            # econv1
            # -----------------
            cur_layer_input = self.encoder_cell_list[0](input[:, :, :, :, t])
            cur_layer_input = self.leakyReLU(cur_layer_input)
            # -----------------
            # ernn1 (layer index 2)
            # -----------------
            h, c = self.encoder_cell_list[1](
                input=cur_layer_input,
                hidden_state=hidden_state[2]   # this is (h_prev, c_prev)
            )
            hidden_state[2] = (h, c)
            cur_layer_input = h
            # -----------------
            # edown1
            # -----------------
            edown = self.encoder_cell_list[2](cur_layer_input)
            edown = self.leakyReLU(edown)
            cur_layer_input = edown
            # -----------------
            # ernn2 (layer index 1)
            # -----------------
            h, c = self.encoder_cell_list[3](
                input=cur_layer_input,
                hidden_state=hidden_state[1]   # this is (h_prev, c_prev)
            )
            hidden_state[1] = (h, c)
            cur_layer_input = h
            # -----------------
            # edown2
            # -----------------
            edown = self.encoder_cell_list[4](cur_layer_input)
            edown = self.leakyReLU(edown)
            cur_layer_input = edown
            # -----------------
            # ernn3 (layer index 0)
            # -----------------
            h, c = self.encoder_cell_list[5](
                input=cur_layer_input,
                hidden_state=hidden_state[0]
            )
            hidden_state[0] = (h, c)
            last_state_list.append(h)

        return last_state_list.pop(), hidden_state

    def _init_hidden(self, batch_size):
        init_states = []
        for i in [1,3,5]:
            init_states.append(self.encoder_cell_list[i].init_hidden(batch_size, device=self.device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param