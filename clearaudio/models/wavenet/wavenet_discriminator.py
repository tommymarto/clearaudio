import torch
import torch.nn as nn
import torch.nn.functional as F


class ZDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_classes = cfg.dataset.number_of_training_eq

        convs = []
        for i in range(cfg.trainer.discriminator.layers):
            in_channels = cfg.trainer.latent_d if i == 0 else cfg.trainer.discriminator.channels
            convs.append(nn.Conv1d(in_channels, cfg.trainer.discriminator.channels, 1))
            convs.append(nn.ELU())
        convs.append(nn.Conv1d(cfg.trainer.discriminator.channels, self.n_classes, 1))

        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout(p=cfg.trainer.discriminator.p_dropout_discriminator)
        #self.linear = nn.Linear(75,1) #75 only valid for the size of input 3sec input

    def forward(self, z):
        z = self.dropout(z)
        logits = self.convs(z)  # (N, n_classes, L)
        # logits = self.linear(F.relu(logits)).squeeze(2)
        logits = logits.mean(2)
        return logits