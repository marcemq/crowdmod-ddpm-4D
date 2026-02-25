import torch
from models.convGRU.convGRUCell import ConvGRUCell
from models.convGRU.encoder import Encoder
from torch import nn
from torch.autograd import Variable

class Forecaster(nn.Module):
    def __init__(self, input_size, input_channels, enc_hidden_channels, forc_hidden_channels, enc_kernels, forc_kernels, device, bias=True):
        super(Forecaster, self).__init__()

        # Make sure that both `forc_kernels` and `hidden_dim` are lists having len == num_layers
        if not len(forc_kernels) == len(forc_hidden_channels) == len(forc_kernels):
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_channels = input_channels
        self.hidden_channels = forc_hidden_channels
        self.enc_hidden_channels = enc_hidden_channels
        self.enc_kernels = enc_kernels
        self.forc_kernels = forc_kernels
        self.bias = bias
        self.device = device
        self.leakyReLU =nn.LeakyReLU(0.2)
        self.encoder = Encoder(input_size=(self.height, self.width), input_channels=self.input_channels, hidden_channels=self.enc_hidden_channels, enc_kernels=self.enc_kernels, num_layers=len(self.enc_kernels), device=device, bias=self.bias)

        forecaster_cell_list = []
        two_stride=2

        # Forecaster layers
        forecaster_cell_list.append(ConvGRUCell(input_size=(self.height//(two_stride*2), self.width//(two_stride*2)),
                                                input_dim=self.hidden_channels[0],
                                                hidden_dim=self.hidden_channels[1],
                                                kernel_size=self.forc_kernels[0],
                                                bias=self.bias))
        forecaster_cell_list.append(nn.ConvTranspose2d(in_channels=self.hidden_channels[1],
                                                       out_channels=self.hidden_channels[2],
                                                       kernel_size=self.forc_kernels[1],
                                                       padding=1,
                                                       stride=two_stride,
                                                       bias=self.bias))
        forecaster_cell_list.append(ConvGRUCell(input_size=(self.height//two_stride, self.width//two_stride),
                                                input_dim=self.hidden_channels[2],
                                                hidden_dim=self.hidden_channels[3],
                                                kernel_size=self.forc_kernels[2],
                                                bias=self.bias))
        forecaster_cell_list.append(nn.ConvTranspose2d(in_channels=self.hidden_channels[3],
                                                       out_channels=self.hidden_channels[4],
                                                       kernel_size=self.forc_kernels[3],
                                                       padding=1,
                                                       stride=two_stride,
                                                       bias=self.bias))
        forecaster_cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                                input_dim=self.hidden_channels[4],
                                                hidden_dim=self.hidden_channels[5],
                                                kernel_size=self.forc_kernels[4],
                                                bias=self.bias))
        forecaster_cell_list.append(nn.Conv2d(in_channels=self.hidden_channels[5],
                                              out_channels=self.hidden_channels[6],
                                              kernel_size=self.forc_kernels[5],
                                              padding=1,
                                              bias=self.bias))
        forecaster_cell_list.append(nn.Conv2d(in_channels=self.hidden_channels[6],
                                              out_channels=self.input_channels,
                                              kernel_size=self.forc_kernels[6],
                                              padding=1,
                                              bias=self.bias))
        
        self.forecaster_cell_list = nn.ModuleList(forecaster_cell_list)

    def forward(self, x_obs, target_obs, teacher_forcing=False, hidden_state=None):
        """
        :param input_tensor (b, h, w, c, t) 
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        # Implement stateful ConvGRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=target_obs.size(0))
            for i in range(len(hidden_state)):
                hidden_state[i] = hidden_state[i].to(self.device)

        target_obs_seq_len = target_obs.size(4)
        forcasted_frames=[]

        for t in range(target_obs_seq_len):
            cur_layer_input, hidden_state = self.encoder(x_obs, hidden_state)
            # frnn1
            h = hidden_state[0]
            h = self.forecaster_cell_list[0](input=cur_layer_input, hidden_state=h) # (b,c,h,w)
            cur_layer_input = h
            hidden_state[0] = h
            # fup1
            fup = self.forecaster_cell_list[1](cur_layer_input)
            fup = self.leakyReLU(fup)
            cur_layer_input = fup
            # frnn2
            h = hidden_state[1]
            h = self.forecaster_cell_list[2](input=cur_layer_input, hidden_state=h) # (b,c,h,w)
            cur_layer_input = h
            hidden_state[1] = h
            #fup2
            fup = self.forecaster_cell_list[3](cur_layer_input)
            fup = self.leakyReLU(fup)
            cur_layer_input = fup
            # frnn3
            h = hidden_state[2]
            h = self.forecaster_cell_list[4](input=cur_layer_input, hidden_state=h) # (b,c,h,w)
            cur_layer_input = h
            hidden_state[2] = h
            # fconv4
            cur_layer_input = self.forecaster_cell_list[5](cur_layer_input)
            cur_layer_input = self.leakyReLU(cur_layer_input)
            # fconv5
            forcasted_next_frame = self.forecaster_cell_list[6](cur_layer_input)
            # append forcasted frames
            forcasted_frames.append(forcasted_next_frame)
            # append next
            if teacher_forcing:
                last_frame = target_obs[:, :, :, :, t]
            else:
                # Apply exp() to density and variance channels only
                channels_to_exp = [0, 3]
                last_frame = forcasted_next_frame.clone()
                last_frame[:, channels_to_exp, :, :] = torch.exp(last_frame[:, channels_to_exp, :, :])

            x_obs = torch.cat((x_obs[:, :, :, :, 1:], last_frame.unsqueeze(4)), dim=4)

        return torch.stack(forcasted_frames, dim=-1)

    def _init_hidden(self, batch_size):
        init_states = []
        for i in [0,2,4]:
            init_states.append(self.forecaster_cell_list[i].init_hidden(batch_size))
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