import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def FFT_for_Period(x, k=2):
    """
    time --> frequency domain,
    amplitude가 큰 top k개 주파수 추출 --> 주기 산출
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k) # rtn values, indices
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block_V1(nn.Module):
    """
    다양한 커널을 이용하여 여러 특징 추출 후 mean
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels  # default: 6
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    """
    FFT를 이용해 top_k개의 주기 추출,
    주기를 이용하여 1D-->2D 변환 후 주기내 패턴, 주기간의 패턴 추출
    Inception_Block_V1: 여러 커널 사이즈를 이용해 Conv 진행(다양한 특징 추출 효과)
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.top_k = configs.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.enc_in, configs.d_model,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_model, configs.enc_in,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()

        ## top k개 주기 추출 및 선정된 각 주기의 진폭 평균값 추출
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        ## 각 주기로 sequence length 분할: 1D --> 2D 변환
        res = []
        for i in range(self.top_k):
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
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()   # (B, len/주기, 주기, N) --> (B, N, len/주기, 주기)

            # conv를 통해 주기내, 주기간의 패턴 특징 추출
            out = self.conv(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])  # padding 부분 제거

        ## 각 주기의 진폭 평균값이 클수록 큰 가중치를 부여하여 특징을 강하게 반영
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])  ## default: 2
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.enc_in)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.linear = nn.Linear(configs.enc_in * configs.seq_len, configs.c_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_enc):
        # TimesNet
        for i in range(self.layer):
            x_enc = self.layer_norm(self.model[i](x_enc))

        output = self.act(x_enc)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.linear(output)  # (batch_size, num_classes)
        return self.sigmoid(output)


