import torch.nn as nn
import torch.nn.functional as F


class DilatedResConv(nn.Module):
    def __init__(self, channels, dilation=1, activation="relu", padding=1, kernel_size=3, left_pad=0):
        super().__init__()
        in_channels = channels

        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()],
                ['swish', nn.SiLU()],
                ['mish', nn.Mish()],
                ['glu', nn.GLU(dim=1)]
                
        ])

        self.left_pad = left_pad
        
        self.dilated_conv = nn.Conv1d(
            in_channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=dilation * padding,
            dilation=dilation,
            bias=True,
        )

        self.activation =  self.activations[activation]
        # if activation == "relu":
        #     self.activation = nn.ReLU() #lambda *args, **kwargs: F.relu(*args, **kwargs, inplace=True)
        # elif activation == "tanh":
        #     self.activation = nn.Tanh()
        # elif activation == "glu":
        if activation == 'glu':
            in_channels = channels // 2

        self.conv_1x1 = nn.Conv1d(in_channels, channels, kernel_size=1, bias=True)

    def forward(self, input):
        x = input

        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        x = self.dilated_conv(x)
        x = self.activation(x)
        x = self.conv_1x1(x)
        return input + x


class WavenetEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_blocks = cfg.trainer.encoder.blocks
        self.n_layers = cfg.trainer.encoder.layers
        self.channels = cfg.trainer.encoder.channels
        self.latent_channels = cfg.trainer.latent_d
        self.activation = cfg.trainer.encoder.activation
        

        try:
            self.encoder_pool = cfg.trainer.encoder.pool
        except AttributeError:
            self.encoder_pool = 800

        layers = []
 
        
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2 ** i
                layers.append(DilatedResConv(self.channels, dilation, self.activation))
        self.dilated_convs = nn.Sequential(*layers)

        self.start = nn.Conv1d(1, self.channels, kernel_size=3, stride=1, padding=1)
        self.conv_1x1 = nn.Conv1d(self.channels, self.latent_channels, 1)
        self.pool = nn.AvgPool1d(self.encoder_pool)

    def receptive_field(self):
        rf = 1
        for _ in range(self.blocks):
            for l in range(self.layer_num):
                dilation = 2 ** l
                rf += (self.kernel_size - 1) * dilation
        return rf

    def forward(self, x):
        x = x / 255 - 0.5
        if x.dim() < 3:
            x = x.unsqueeze(1)

        x = self.start(x)
        x = self.dilated_convs(x)
        x = self.conv_1x1(x)
        x = self.pool(x)
        return x
