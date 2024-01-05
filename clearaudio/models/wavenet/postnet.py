import torch.nn as nn
import torch



class PostNet(nn.Module):
    def __init__(self,cfg,):
        super(PostNet, self).__init__()
        
        self.postnet = torch.nn.ModuleList()

        activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU(negative_slope=cfg.trainer.postnet.neg_slope_lrelu)],
                ['relu', nn.ReLU()],
                ['tanh', nn.Tanh()],
                ['swish', nn.SiLU()],
                ['mish', nn.Mish()]
        ])
        for layer in range(cfg.trainer.postnet.n_layers-1) :
            input_channels = cfg.trainer.postnet.input_dim if layer == 0 else cfg.trainer.postnet.n_channels
            output_channels = cfg.trainer.postnet.output_dim if layer == cfg.trainer.postnet.n_layers-1 else cfg.trainer.postnet.n_channels 

            self.postnet.append(torch.nn.Sequential(
                                torch.nn.Conv1d(
                                    input_channels,
                                    output_channels,
                                    cfg.trainer.postnet.kernel_size,
                                    stride = cfg.trainer.postnet.stride,
                                    padding = "same",
                                    bias = False
                                ),
                                activations[cfg.trainer.postnet.activation],
                                torch.nn.Dropout(cfg.trainer.postnet.dropout_rate)
                )
            )

        self.postnet.append(torch.nn.Sequential(
                            torch.nn.Conv1d(
                                output_channels,
                                cfg.trainer.postnet.output_dim,
                                cfg.trainer.postnet.kernel_size,
                                stride = cfg.trainer.postnet.stride,
                                padding = "same",
                                bias = False
                            ),
                        nn.Tanh())
        )
    def forward(self, x):
        for layer in self.postnet:          
           x = layer(x) 
        return x
