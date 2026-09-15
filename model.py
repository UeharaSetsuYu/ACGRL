import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads= 4, ff_hidden_dim=256, dropout=0.1):
        super().__init__() 
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
 
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
 
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None): 
        # ---- Self-Attention ----
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # ---- Feed Forward ----
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights

class Encoder_Consistency(nn.Module):
    def __init__(self, auto_dim, activa = 'relu', batchnorm = True, LayerNorm = False, dp = 0.2):
        super(Encoder_Consistency, self).__init__()
        self._dim = len(auto_dim) - 1
        self._activation = activa
        self.batchnorm = batchnorm
        self.LayerNorm = LayerNorm
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(nn.Linear(auto_dim[i], auto_dim[i + 1]))
            if i < self._dim - 1:
                if self.batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(auto_dim[i + 1]))
                    encoder_layers.append(nn.Dropout(dp))
                elif self.LayerNorm:
                    encoder_layers.append(nn.LayerNorm(auto_dim[i + 1]))
                    encoder_layers.append(nn.Dropout(dp))

                if self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                elif self._activation == 'elu':
                    encoder_layers.append(nn.ELU())
                elif self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'LeakyRelu':
                    encoder_layers.append(nn.LeakyReLU())
                elif self._activation == 'GELU':
                    encoder_layers.append(nn.GELU())
                else:
                    raise ValueError('Activation function is not supported')

        self.encoder_c = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder_c(x)


class Encoder_Complementarity(nn.Module):
    def __init__(self, auto_dim, activa='relu', batchnorm=True, LayerNorm=False, dp = 0.2):
        super(Encoder_Complementarity, self).__init__()
        self._dim = len(auto_dim) - 1
        self._activation = activa
        self.batchnorm = batchnorm
        self.LayerNorm = LayerNorm
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(nn.Linear(auto_dim[i], auto_dim[i + 1]))
            if i < self._dim - 1:
                if self.batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(auto_dim[i + 1]))
                    encoder_layers.append(nn.Dropout(dp))
                elif self.LayerNorm:
                    encoder_layers.append(nn.LayerNorm(auto_dim[i + 1]))
                    encoder_layers.append(nn.Dropout(dp))

                if self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                elif self._activation == 'elu':
                    encoder_layers.append(nn.ELU())
                elif self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'LeakyRelu':
                    encoder_layers.append(nn.LeakyReLU())
                elif self._activation == 'GELU':
                    encoder_layers.append(nn.GELU())
                else:
                    raise ValueError('Activation function is not supported')

        self.encoder_s = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder_s(x)

class Decoder_Comprehensive(nn.Module):
    def __init__(self, auto_dim, activa='relu', batchnorm=True, LayerNorm=False, dp = 0.2):
        super(Decoder_Comprehensive, self).__init__()
        self._dim = len(auto_dim) - 1
        self._activation = activa
        self.batchnorm = batchnorm
        self.LayerNorm = LayerNorm
        decoder_layers = []
        decoder_dim = [i for i in reversed(auto_dim)]
        for i in range(self._dim):
            if i == 0:
                decoder_layers.append(nn.Linear(decoder_dim[i] * 2, decoder_dim[i + 1]))
            else:
                decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if i < self._dim - 1:
                if self.batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
                    decoder_layers.append(nn.Dropout(dp))
                elif self.LayerNorm:
                    decoder_layers.append(nn.LayerNorm(decoder_dim[i + 1]))
                    decoder_layers.append(nn.Dropout(dp))
                if self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                elif self._activation == 'elu':
                    decoder_layers.append(nn.ELU())
                elif self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'LeakyRelu':
                    decoder_layers.append(nn.LeakyReLU())
                else:
                    raise ValueError('Activation function is not supported')

        self.decoder = nn.Sequential(*decoder_layers)
    def forward(self, x):
        return self.decoder(x)

class MLP_shared(nn.Module):
    def __init__(self, embed_dim):
        super(MLP_shared, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
                                  nn.LayerNorm(embed_dim // 2),
                                  nn.Dropout(0.2),
                                  nn.Linear(embed_dim // 2, embed_dim//2), nn.ReLU(),
                                  nn.LayerNorm(embed_dim // 2),
                                  nn.Dropout(0.2),
                                  nn.Linear(embed_dim//2, embed_dim)) # version 2
        self.norm1 = nn.LayerNorm(embed_dim)
        self.atten = AttentionBlock(embed_dim, num_heads=4, ff_hidden_dim=embed_dim * 4, dropout=0.2)   # Ngs:0.5

        self.SE = SEBlock(embed_dim, reduction = 2)
    def forward(self, x): 
        x = self.norm1(x + self.SE(x))
        x = x.unsqueeze(0)
        attn, _ = self.atten(x)
        attn = attn.squeeze(0)
        return self.net1(attn)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),   
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  
        )

    def forward(self, x): 
        se = torch.mean(x, dim=0, keepdim=True)   
        se = self.se(se)   
        return x * se   
class MLP_Pseudo(nn.Module):
    def __init__(self, embed_dim, n_cluster, dp = 0.2):
        super(MLP_Pseudo, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
                                  nn.Dropout(dp),
                                  nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
                                  nn.Linear(embed_dim // 2, n_cluster))

    def forward(self, x):
        x = self.net1(x)
        probs = F.softmax(x, dim=-1)
        return probs
class Discriminator(nn.Module):
    def __init__(self, embed_dim, view, dp = 0.2):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, embed_dim  ),
                                 nn.LeakyReLU(0.2), #
                                 nn.Dropout(dp),
                                 nn.Linear(embed_dim  , embed_dim // 2),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(embed_dim // 2, view+1)
                                 ) 
    def forward(self, x):
        probs = self.net(x)
        return probs

class ViewDiscriminator(nn.Module):
    def __init__(self, embed_dim, K, dp = 0.2):   # embed_dim is
        super(ViewDiscriminator, self).__init__()
        
        self.net_SpectralNorm = nn.Sequential(
                                 spectral_norm(nn.Linear(embed_dim, embed_dim)),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(dp),
                                 spectral_norm(nn.Linear(embed_dim, embed_dim // 2)),
                                 nn.LeakyReLU(0.2),
                                 spectral_norm(nn.Linear(embed_dim // 2, K))
                               )


    def forward(self, x):
        probs = self.net_SpectralNorm(x) 
        return probs



class Filter(nn.Module):
    def __init__(self, embed_dim):
        super(Filter, self).__init__()
        self.SE = SEBlock(embed_dim, reduction=2)
        self.atten = AttentionBlock(embed_dim, num_heads=4, ff_hidden_dim=embed_dim * 4, dropout=0.5)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.net1 = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim//2),
                            nn.ReLU(),
                            nn.LayerNorm(embed_dim // 2),
                            nn.Linear(embed_dim//2, embed_dim)
                                  )    # version 2
        self.norm1 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        se = self.norm1(x + self.SE(x))
        attn_tmp, _ = self.atten(se.unsqueeze(0))
        attn = attn_tmp.squeeze(0)
        res = self.net1(attn)
        alpha = torch.sigmoid(self.alpha)
        return alpha * se + (1 - alpha) * res

class ACGRL(nn.Module):
    def __init__(self, auto_dim, view_num, cluster_n, embed_dim = 128, activa = 'relu', batchnorm = False, layernorm = True, dp = 0.5):
        super(ACGRL, self).__init__()
        self.cluster_n = cluster_n
        self.view_num = view_num
        self.embed_dim = embed_dim
 
        self.MLP = MLP_shared(embed_dim)
        self.filter = nn.ModuleList([Filter(embed_dim) for _ in range(view_num)])
        
        self.pseudo_mlp = nn.ModuleList([MLP_Pseudo(embed_dim, self.cluster_n, dp = 0.2) for _ in range(view_num)])
        self.view_discriminator = ViewDiscriminator(embed_dim, view_num, dp = 0.2)
        self.discriminator = Discriminator(embed_dim, view_num, dp = 0.2)
        
        self.Encoder_c = nn.ModuleList([Encoder_Consistency(dim, activa=activa, batchnorm=batchnorm, LayerNorm=layernorm, dp = dp) for dim in auto_dim])
        self.Encoder_s = nn.ModuleList([Encoder_Complementarity(dim, activa=activa, batchnorm=batchnorm, LayerNorm=layernorm, dp = dp) for dim in auto_dim])
        self.Decoder = nn.ModuleList([Decoder_Comprehensive(dim, activa=activa, batchnorm=batchnorm, LayerNorm=False, dp = dp) for dim in auto_dim])

    def forward(self, x):
        x_hat_list, pseudo_list, z_con, z_share = [], [], [], []
        for view in range(self.view_num):
             

            share_latent = self.Encoder_c[view](x[view])
            specific_latent = self.Encoder_s[view](x[view])
 

            z_specific = self.filter[view](specific_latent)
            share_z = self.MLP(share_latent)

            latent = torch.cat([z_specific, share_z], dim=1)
            x_hat_list.append(self.Decoder[view](latent))
            pseudo_list.append(self.pseudo_mlp[view](latent))

            z_con.append(z_specific)
            z_share.append(share_z)

        return x_hat_list, pseudo_list, z_con, z_share


