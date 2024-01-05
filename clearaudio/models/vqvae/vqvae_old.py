import numpy as np
import torch as t
import torch.nn as nn
import ast

from clearaudio.models.vqvae.encdec import Encoder, Decoder, assert_shape
from clearaudio.models.vqvae.bottleneck import NoBottleneck, Bottleneck

class VQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.levels = cfg.trainer.vqvae.levels
        self.cfg = cfg
        self.nb_blocks = ast.literal_eval(cfg.trainer.vqvae.nb_blocks)
        self.strides_t = ast.literal_eval(cfg.trainer.vqvae.strides_t)

        encoder = lambda level: Encoder(cfg, levels= level + 1, nb_blocks= self.nb_blocks[:level+1], strides_t= self.strides_t[:level+1], width = None, depth = None)
        decoder = lambda level: Decoder(cfg, levels= level + 1, nb_blocks= self.nb_blocks[:level+1], strides_t= self.strides_t[:level+1], width = None, depth = None)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(self.levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        if cfg.trainer.codebook.use_bottleneck:
            self.bottleneck = Bottleneck(cfg)
        else:
            self.bottleneck = NoBottleneck(cfg)

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def forward(self, x):

        # Encode/Decode
        x_out_encoder = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x)
            x_out_encoder.append(x_out[-1])

        latent_quantised , xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(x_out_encoder)
        fit_loss = quantiser_metrics[0]
        print(fit_loss)
        commit_loss = sum(commit_losses)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)
            x_outs.append(x_out)


        return x_outs, x_out_encoder, latent_quantised, xs_quantised, commit_loss, fit_loss