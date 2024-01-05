import torch
import torch.nn as nn
from clearaudio.models.transformer.pos_emb import get_1d_sincos_pos_embed
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname:
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)


class Attention(nn.Module):
    """
    Simple Self-Attention algorithm. Potential for optimization using a non-quadratic attention mechanism in complexity.
    -> Linformer, Reformer etc.
    """
    def __init__(self, dim=768, heads=8):
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.norm, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention, splitting it up to multiple Self-Attention layers and concatenating
    the results and subsequently running it through one linear layer of same dimension.
    """
    def __init__(self, dim=768, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.self_attention_heads = nn.ModuleList([Attention(dim, heads) for _ in range(heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(self, x):
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out


class Encoder(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """
    def __init__(self, dim=768, hidden_dim=3072):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim)
        self.LayerNorm2 = nn.LayerNorm(dim)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.GELU()
        ])

    def forward(self, x):
        attn = self.MultiHeadAttention(x)
        x = x.add(attn)
        x = self.LayerNorm1(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.LayerNorm2(x)
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, cfg):
        super(BidirectionalTransformer, self).__init__()

        self.temporal_dimension = cfg.trainer.timesteps / cfg.trainer.vqvae.stride_t**cfg.trainer.vqvae.nb_blocks
        assert self.temporal_dimension == int(self.temporal_dimension)
        self.temporal_dimension = int(self.temporal_dimension)
        self.first_conv = nn.Conv1d(cfg.trainer.codebook.emb_channels, cfg.trainer.transformer.hidden_dim, 31, padding='same')
        self.pos_embed = nn.Parameter(torch.zeros(1, self.temporal_dimension, cfg.trainer.transformer.hidden_dim)).detach() #batch, T , Channels
        self.EncoderLayers = nn.ModuleList([Encoder(cfg.trainer.transformer.hidden_dim) for _ in range(cfg.trainer.transformer.N)])

        # self.Token_Prediction = nn.Linear(in_features=cfg.trainer.transformer.hidden_dim, out_features=cfg.trainer.codebook.nb_bins)
        self.projection = nn.Linear(in_features= cfg.trainer.transformer.hidden_dim, out_features=cfg.trainer.decoder.in_channels)
        self.apply(weights_init)
        pos_embed = get_1d_sincos_pos_embed(cfg.trainer.transformer.hidden_dim, np.arange(self.temporal_dimension))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):

        # token_embeddings = self.tok_emb(x)
        x_embeddings = self.first_conv(x)
        x_embeddings = x_embeddings.transpose(1,2)
        device = x_embeddings.get_device()
        self.pos_embed = self.pos_embed.to(device)
        # t = token_embeddings.shape[1]
        embed = x_embeddings + self.pos_embed
        for enc_layer in self.EncoderLayers:
            embed = enc_layer(embed)
        # tkn_prd = self.Token_Prediction(embed)
        proj_x = self.projection(embed)
        return proj_x.transpose(1,2)