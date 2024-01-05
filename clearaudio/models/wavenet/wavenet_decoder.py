import torch
import torch.nn as nn
import torch.nn.functional as F
from clearaudio.models.wavenet.postnet import PostNet


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class WavenetLayer(nn.Module):
    def __init__(self, residual_channels, 
                skip_channels, cond_channels, kernel_size=2, dilation=1,
                use_encoder=True, causal=True, activation : str = 'original',
                batch_norm=False):
        super(WavenetLayer, self).__init__()
        self.nb_chunk = 0
        self.activation_layer = []
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()],
                ['swish', nn.SiLU()],
                ['mish', nn.Mish()],
                ['original', None]
                
        ])

        if activation == 'original':
            self.activation_layer = [nn.Sigmoid(), nn.Tanh()]
            self.nb_chunk = 2
        else:
            for activation_fc in activation.split(','):
                self.nb_chunk += 1
                self.activation_layer.append(self.activations[activation_fc.strip()])
        if causal:
            self.causal = CausalConv1d(
                residual_channels, 2 * residual_channels, kernel_size, dilation=dilation, bias=True
            )
        else:
            if not (kernel_size % 2):
                self.causal = nn.Conv1d(
                    residual_channels,
                    self.nb_chunk * residual_channels,
                    kernel_size + 1,
                    padding="same",
                    dilation=dilation,
                    bias=True,
                )
            else:
                self.causal = nn.Conv1d(
                    residual_channels, self.nb_chunk * residual_channels, kernel_size, padding="same", dilation=dilation, bias=True
                )
        if use_encoder:
            self.condition = nn.Conv1d(cond_channels, self.nb_chunk * residual_channels, kernel_size=1, bias=True)
        if batch_norm:
            self.batch_norm = nn.SyncBatchNorm(self.nb_chunk * residual_channels)
        else:
            self.batch_norm = None
        self.residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1, bias=True)
        self.skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1, bias=True)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    def forward(self, x, c=None):
        x = self.causal(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if c is not None:
            x = self._condition(x, c, self.condition)
        assert x.size(1) % self.nb_chunk == 0

        chunks = x.chunk(self.nb_chunk, 1)
        x = torch.ones_like(chunks[0])
        for i, subchunk in enumerate(chunks):
            x = self.activation_layer[i](subchunk) * x
            
        # gate = torch.sigmoid(gate)
        # output = torch.tanh(output)
        # x = gate * output

        residual = self.residual(x)
        skip = self.skip(x)
        return residual, skip


class WavenetDecoder(nn.Module):
    def __init__(self, cfg, create_layers=True, shift_input=False, channels_in=1, classes=256, causal=False):
        super().__init__()

        self.blocks = cfg.trainer.decoder.blocks
        self.layer_num = cfg.trainer.decoder.layers
        self.kernel_size = cfg.trainer.decoder.kernel_size
        self.skip_channels = cfg.trainer.decoder.skip_channels
        self.residual_channels = cfg.trainer.decoder.residual_channels
        self.cond_channels = cfg.trainer.latent_d
        self.classes = classes
        self.shift_input = shift_input
        self.channels_in = channels_in
        self.causal = causal
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU(negative_slope=cfg.trainer.decoder.neg_slope_lrelu)],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()],
                ['swish', nn.SiLU()],
                ['mish', nn.Mish()],
                ['original', None]
                
        ])

        self.last_activation = self.activations[cfg.trainer.decoder.last_activation]
        if create_layers:
            layers = []
            for _ in range(self.blocks):
                for i in range(self.layer_num):
                    dilation = 2 ** i
                    layers.append(
                        WavenetLayer(
                            self.residual_channels,
                            self.skip_channels,
                            self.cond_channels,
                            self.kernel_size,
                            dilation,
                            use_encoder=cfg.trainer.encoder.active,
                            causal=self.causal,
                            activation=cfg.trainer.decoder.layer_activation,
                            batch_norm=cfg.trainer.decoder.batch_norm_active
                        )
                    )
            self.layers = nn.ModuleList(layers)

        if self.causal:
            self.first_conv = CausalConv1d(self.channels_in, self.residual_channels, kernel_size=self.kernel_size)
        elif not (self.kernel_size % 2):
            self.first_conv = nn.Conv1d(
                self.channels_in, self.residual_channels, kernel_size=self.kernel_size + 1, padding="same")
        else:
            self.first_conv = nn.Conv1d(
                self.channels_in, self.residual_channels, kernel_size=self.kernel_size, padding="same")

        self.skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
        if cfg.trainer.encoder.active:
            self.condition = nn.Conv1d(self.cond_channels, self.skip_channels, kernel_size=1)

        self.fc = nn.Conv1d(self.skip_channels * 2, self.skip_channels, kernel_size=1)
        self.logits = nn.Conv1d(self.skip_channels, self.classes, kernel_size=1)


    def receptive_field(self):
        rf = 1
        for _ in range(self.blocks):
            for l in range(self.layer_num):
                dilation = 2 ** l
                rf += (self.kernel_size - 1) * dilation
        return rf

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    @staticmethod
    def _upsample_cond(x, c):
        bsz, channels, length = x.size()
        cond_bsz, cond_channels, cond_length = c.size()
        assert bsz == cond_bsz
        if c.size(2) != 1:
            try:
                assert length % cond_length == 0
                c = c.unsqueeze(3).repeat(1, 1, 1, length // cond_length)
                c = c.view(bsz, cond_channels, length)
            except AssertionError:
                print(length, cond_length, length % cond_length)
                raise NotImplementedError
        return c

    @staticmethod
    def shift_right(x):
        x = F.pad(x, (1, 0))
        return x[:, :, :-1]
    
    def forward(self, x, c=None):
        if x.dim() < 3:
            x = x.unsqueeze(1)
        if (not "Half" in x.type()) and (not "Float" in x.type()):
            x = x.float()

        if self.shift_input:
            x = self.shift_right(x)

        if c is not None:
            c = self._upsample_cond(x, c)

        residual = self.first_conv(x)
        skip = self.skip_conv(residual)

        for layer in self.layers:
            r, s = layer(residual, c)
            residual = residual + r
            skip = skip + s
        skip = skip + residual  ##### TODO pas agréable mais distributed pas content
        skip = torch.cat((skip, residual),1)
        skip = self.last_activation(skip)
        skip = self.fc(skip)
        if c is not None:
            skip = self._condition(skip, c, self.condition)
        skip = self.last_activation(skip)
        logits = self.logits(skip)

        return logits

