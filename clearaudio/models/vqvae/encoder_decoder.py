import torch as t
import torch.nn as nn
from clearaudio.models.vqvae.resnet import Resnet, Resnet1D
from clearaudio.utils.utils import assert_shape

class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, nb_blocks,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False):
        super().__init__()
        blocks = []
        kernel_size, pad_t = stride_t * 2, stride_t // 2
        if nb_blocks > 0:
            for i in range(nb_blocks):
                block = nn.Sequential(
                    nn.Conv1d(input_emb_width if i == 0 else width, width, kernel_size, stride_t, pad_t),
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, nb_blocks,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, 
                 reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        if nb_blocks > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(nb_blocks):
                block = nn.Sequential(
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation, checkpoint_res=checkpoint_res),
                    nn.ConvTranspose1d(width, input_emb_width if i == (nb_blocks - 1) else width, filter_t, stride_t, pad_t)
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.input_channels = cfg.trainer.vqvae.input_channels
        self.emb_channels = cfg.trainer.vqvae.emb_channels
        self.levels = cfg.trainer.vqvae.levels
        self.nb_blocks = cfg.trainer.encoder.nb_blocks
        self.strides_t = cfg.trainer.encoder.strides_t
        self.width = cfg.trainer.encoder.width
        self.depth = cfg.trainer.encoder.depth
        self.m_conv = cfg.trainer.encoder.m_conv
        self.dilation_growth_rate = cfg.trainer.encoder.dilation_growth_rate
        self.dilation_cycle = cfg.trainer.encoder.dilation_cycle
        self.zero_out = cfg.trainer.encoder.zero_out
        self.res_scale = cfg.trainer.encoder.res_scale
        self.reverse_decoder_dilation = cfg.trainer.encoder.reverse_decoder_dilation
        self.checkpoint_res = cfg.trainer.encoder.checkpoint_res

        level_block = lambda level, nb_block, stride_t: EncoderConvBlock(self.input_channels if level == 0 else self.emb_channels,
                                                           self.emb_channels, nb_block, stride_t, width = self.width, depth = self.depth, m_conv = self.m_conv, 
                                                           dilation_growth_rate = self.dilation_growth_rate, dilation_cycle = self.dilation_cycle,
                                                           zero_out = self.zero_out, res_scale = self.res_scale)

        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), self.nb_blocks, self.strides_t)
        for level, nb_block, stride_t in iterator:
            self.level_blocks.append(level_block(level, nb_block, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, nb_block, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** nb_block)
            assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs

class Decoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.input_channels = cfg.trainer.vqvae.input_channels
        self.emb_channels = cfg.trainer.vqvae.emb_channels
        self.levels = cfg.trainer.vqvae.levels
        self.nb_blocks = cfg.trainer.decoder.nb_blocks
        self.strides_t = cfg.trainer.decoder.strides_t     
        self.levels = cfg.trainer.decoder.levels
        self.width = cfg.trainer.decoder.width
        self.depth = cfg.trainer.decoder.depths
        self.m_conv = cfg.trainer.decoder.m_conv
        self.dilation_growth_rate = cfg.trainer.decoder.dilation_growth_rate
        self.dilation_cycle = cfg.trainer.decoder.dilation_cycle
        self.zero_out = cfg.trainer.decoder.zero_out
        self.res_scale = cfg.trainer.decoder.res_scale
        self.checkpoint_res = cfg.trainer.decoder.checkpoint_res

        level_block = lambda level, down_t, stride_t: DecoderConvBock(self.emb_channels, self.emb_channels,
                                                            down_t, stride_t, width = self.width,
                                                            depth = self.depth, m_conv = self.m_conv, 
                                                            dilation_growth_rate = self.dilation_growth_rate, dilation_cycle = self.dilation_cycle,
                                                            zero_out = self.zero_out, res_scale = self.res_scale)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), self.nb_blocks, self.strides_t)
        for level, nb_block, stride_t in iterator:
            self.level_blocks.append(level_block(level, nb_block, stride_t))

        self.out = nn.Conv1d(self.emb_channels, self.input_channels, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.emb_channels
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.nb_blocks, self.strides_t)))
        for level, nb_block, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t ** nb_block)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x