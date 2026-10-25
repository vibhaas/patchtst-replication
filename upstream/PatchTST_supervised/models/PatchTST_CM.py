__all__ = ['PatchTST_CM']

# [replication] Channel-mixing PatchTST for the "P only" ablation (paper Table 7 / Appendix A.4.3).
# Not part of the official code. Instead of reshaping B x M x P x N to (B*M) x P x N (channel-independent),
# each token holds the same patch of every channel: B x (M*P) x N. With patch_len = stride = 1 this is the
# "Original" (no patching, channel-mixing) variant.

import torch
from torch import nn

from layers.PatchTST_backbone import TSTEncoder
from layers.PatchTST_layers import positional_encoding
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_vars = configs.enc_in
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        d_model = configs.d_model

        self.revin = configs.revin
        if self.revin: self.revin_layer = RevIN(self.n_vars, affine=configs.affine, subtract_last=configs.subtract_last)

        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        self.padding_patch = configs.padding_patch
        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            patch_num += 1

        self.W_P = nn.Linear(self.n_vars * self.patch_len, d_model)        # one token = one patch of all channels
        self.W_pos = positional_encoding('zeros', True, patch_num, d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.encoder = TSTEncoder(patch_num, d_model, configs.n_heads, d_ff=configs.d_ff, norm='BatchNorm',
                                  dropout=configs.dropout, activation='gelu', res_attention=True, n_layers=configs.e_layers)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Dropout(configs.head_dropout),
                                  nn.Linear(d_model * patch_num, self.n_vars * self.pred_len))

    def forward(self, x):                                                   # x: [bs x seq_len x nvars]
        bs = x.shape[0]
        if self.revin: x = self.revin_layer(x, 'norm')
        z = x.permute(0, 2, 1)                                              # z: [bs x nvars x seq_len]
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 2, 1, 3).reshape(bs, z.shape[2], -1)               # z: [bs x patch_num x (nvars * patch_len)]
        z = self.dropout(self.W_P(z) + self.W_pos)                          # z: [bs x patch_num x d_model]
        z = self.encoder(z)                                                 # z: [bs x patch_num x d_model]
        z = self.head(z.permute(0, 2, 1))                                   # z: [bs x nvars * pred_len]
        z = z.reshape(bs, self.n_vars, self.pred_len).permute(0, 2, 1)      # z: [bs x pred_len x nvars]
        if self.revin: z = self.revin_layer(z, 'denorm')
        return z
