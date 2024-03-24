import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    # print('FFT for period>>')
    xf = torch.fft.rfft(x, dim=1)
    # print(x.shape, xf.shape)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    # print(abs(xf).shape, abs(xf).mean(0).shape, abs(xf).mean(0).mean(-1).shape)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k) # rtn values, indices
    # print('top list: ', top_list)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    # print(x.shape[1], top_list, period)
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # print(period_weight.shape)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()   # (B, 주기갯수?주파수?, 주기, N) --> (B, N, 주기갯수?주파수?, 주기)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # print(f'out: {out.shape}')
            res.append(out[:, :self.seq_len, :])  # padding 부분 제거
        res = torch.stack(res, dim=-1)
        # print(f'res: {res.shape}')
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        # print(f'period_weight: {period_weight.shape}')
        # print(f'period_weight: {period_weight.unsqueeze(1).shape}')
        # print(f'period_weight: {period_weight.unsqueeze(1).unsqueeze(1).shape}')
        # print(f'period_weight: {period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1).shape}')

        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # print((res * period_weight).shape)
        res = torch.sum(res * period_weight, -1)
        # print(res.shape, x.shape)
        # residual connection
        res = res + x
        # print(res.shape)
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])  ## default: 2
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model * configs.seq_len, configs.c_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        # return output  # [B, N]
        return self.sigmoid(output)


