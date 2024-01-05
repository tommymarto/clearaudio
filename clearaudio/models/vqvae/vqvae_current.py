import torch
import torch.nn as nn
import torch.nn.functional as F

from clearaudio.models.vqvae.resnet import Resnet1D
from clearaudio.models.vqvae.bottleneck import Bottleneck as Codebook
from clearaudio.models.transformer.transformer import BidirectionalTransformer as Transformer

####

class UpSampleBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size = 3, stride = 1, pad = 1, mode="interpolate"):
        super(UpSampleBlock, self).__init__()
        self.scale_factor = stride
        self.mode = mode
        if self.mode == 'interpolate':
            self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, padding = 'same')
        elif self.mode == 'transpose_conv':
            self.conv = nn.ConvTranspose1d(channels_in, channels_out, kernel_size, stride, pad)
        elif self.mode == "pixel_shuffle":
            self.conv = nn.Conv1d(channels_in, channels_in*self.scale_factor**2, 21, padding = 'same')
            self.shuffle = nn.PixelShuffle(self.scale_factor)
        else:
            raise NotImplementedError()

    def forward(self, x):
        device = x.get_device()
        if self.mode == 'interpolate':
            x = F.interpolate(x, scale_factor=self.scale_factor)
            return self.conv(x)
        elif self.mode == "pixel_shuffle":
            # zeros_size = x.size(1)*self.scale_factor**2-x.size(1)*self.scale_factor ### Fill the channels with enough 0 to make a 2-D pixel shuffling
            # zeros = torch.zeros(x.size(0), zeros_size, x.size(2)).detach().to(device)
            # x_prime = torch.cat((self.conv(x), zeros), dim=1).unsqueeze(-1)
            # y = self.shuffle(x_prime)[:,0,:,:].reshape(x.size(0),x.size(1), -1).contiguous()
            x_prime = self.conv(x).unsqueeze(-1)
            y = self.shuffle(x_prime)[:,:,:,0].contiguous()
            return y
        elif self.mode == "transpose_conv":
            return self.conv(x)
        else:
            raise NotImplementedError()


class DownSampleBlock(nn.Module):
    def __init__(self, channels, kernel_size = 3, stride_t = 2, pad_t = 0):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, stride_t, pad_t)        
    def forward(self, x):
        # pad = (0, 1, 0, 1)
        # x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class GroupNorm(nn.Module):
    def __init__(self, in_channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.in_channels = cfg.trainer.vqvae.input_channels
        self.width = cfg.trainer.encoder.width
        self.output_channels = cfg.trainer.codebook.emb_channels
        self.depth = cfg.trainer.encoder.depth
        self.m_conv = cfg.trainer.encoder.m_conv
        self.dilation_growth_rate = cfg.trainer.encoder.dilation_growth_rate
        self.dilation_cycle = cfg.trainer.encoder.dilation_cycle
        self.zero_out = cfg.trainer.encoder.zero_out
        self.res_scale = cfg.trainer.encoder.res_scale
        blocks = []
        stride_t = cfg.trainer.vqvae.stride_t
        blocks.append(nn.Conv1d(self.in_channels, self.width, 32, 1, padding='same'))
        for i in range(cfg.trainer.vqvae.nb_blocks):
            kernel_size, pad_t = stride_t * 2, stride_t // 2
            block = nn.Sequential(
                    DownSampleBlock(self.width, kernel_size, stride_t, pad_t),
                    Resnet1D(self.width, self.depth, self.m_conv, self.dilation_growth_rate, self.dilation_cycle, self.zero_out, self.res_scale),
                )
            blocks.append(block)
        block = nn.Conv1d(self.width, self.output_channels, cfg.trainer.encoder.last_kernel_size, 1, padding='same')
        blocks.append(block)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, cfg, kernel_size = None, output_classes = None):
        super(Decoder, self).__init__()
        self.first_kernel_size = cfg.trainer.decoder.first_kernel_size
        self.in_channels = cfg.trainer.decoder.in_channels
        self.width = cfg.trainer.decoder.width
        self.output_channels = cfg.trainer.vqvae.output_channels
        stride_t = cfg.trainer.vqvae.stride_t
        self.depth = cfg.trainer.decoder.depth
        self.m_conv = cfg.trainer.decoder.m_conv
        self.dilation_growth_rate = cfg.trainer.decoder.dilation_growth_rate
        self.dilation_cycle = cfg.trainer.decoder.dilation_cycle
        self.zero_out = cfg.trainer.decoder.zero_out
        self.res_scale = cfg.trainer.decoder.res_scale
        self.upsampling_mode = cfg.trainer.decoder.upsampling_mode
        

        blocks = []
        if kernel_size is None:
            self.kernel_size, pad_t = stride_t * 2, stride_t // 2
        else:
            self.kernel_size = kernel_size
        block = nn.Conv1d(self.in_channels, self.width, self.first_kernel_size, 1, padding = 'same')
        blocks.append(block)
        for i in range(cfg.trainer.vqvae.nb_blocks):
            block = nn.Sequential(
                    Resnet1D(self.width, self.depth, self.m_conv, self.dilation_growth_rate, self.dilation_cycle, zero_out=self.zero_out, res_scale=self.res_scale),
                    UpSampleBlock(self.width, self.width, self.kernel_size, stride_t, pad_t, mode = self.upsampling_mode)
                )
            blocks.append(block)
        
        blocks.append(GroupNorm(self.width))
        blocks.append(Swish())
        if output_classes is None:
            self.output_classes = cfg.trainer.decoder.classes
        else:
            self.output_classes = output_classes
        blocks.append(nn.Conv1d(self.width, self.output_classes, kernel_size=63, stride=1, padding = 'same'))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class VQVAE(nn.Module):
    def __init__(self, cfg, output_classes = None):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg, output_classes = output_classes)
        self.codebook = Codebook(cfg)
        self.transformer_active = cfg.trainer.vqvae.transformer.active
        if cfg.trainer.vqvae.transformer.active :
            self.transformer = Transformer(cfg)

    def forward(self, x):
        encoded_x= self.encoder(x)
        if torch.isnan(encoded_x).sum():
            print('encoded_x',encoded_x)
        zs, x_quantised, commit_loss, metric = self.codebook(encoded_x)
        if torch.isnan(x_quantised).sum():
            print("x_quantised",x_quantised)
        if self.transformer_active:
            x_quantised = self.transformer(x_quantised)
        decoded_x = self.decoder(x_quantised)

        return decoded_x, zs, commit_loss, metric

    def encode(self, x):
        encoded_x= self.encoder(x)
        zs, x_quantised, commit_loss, metric = self.codebook(encoded_x)
        return zs, x_quantised, commit_loss, metric, encoded_x

    def decode(self, z):
        decoded_music = self.decoder(z)
        return decoded_music
##TODO
    # def calculate_lambda(self, nll_loss, g_loss):
    #     last_layer = self.decoder.model[-1]
    #     last_layer_weight = last_layer.weight
    #     nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
    #     g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

    #     λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    #     λ = torch.clamp(λ, 0, 1e4).detach()
    #     return 0.8 * λ

    # @staticmethod
    # def adopt_weight(disc_factor, i, threshold, value=0.):
    #     if i < threshold:
    #         disc_factor = value
    #     return disc_factor

    # def load_checkpoint(self, path):
    #     self.load_state_dict(torch.load(path))
    #     print("Loaded Checkpoint for VQGAN....")




class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = GroupNorm(in_channels)
        self.q = torch.nn.Conv1d(in_channels, in_channels, 1, 1, 0)
        self.k = torch.nn.Conv1d(in_channels, in_channels, 1, 1, 0)
        self.v = torch.nn.Conv1d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)

        attn = attn.permute(0, 2, 1)
        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        A = self.proj_out(A)

        return x + A



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.channel_up(x)
        else:
            return x + self.block(x)