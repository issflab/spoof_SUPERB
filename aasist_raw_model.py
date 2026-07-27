"""
aasist_raw_model.py
-------------------
Standalone AASIST baseline: sinc-convolution front-end on the raw waveform +
the AASIST spectro-temporal graph-attention back-end. No SSL encoder.

Relationship to aasist_model.py:
  aasist_model.Model is AASIST's graph back-end mounted on an s3prl SSL
  front-end -- that is the benchmark's SSL system, not the AASIST baseline.
  This module reuses the three graph classes from it verbatim (they were
  verified byte-identical to the published AASIST implementation) and supplies
  the two pieces the SSL variant had to drop:

    - CONV        : the sinc band-pass front-end (RawNet2/AASIST), absent from
                    aasist_model.py because SSL features replace it.
    - Residual_block : re-declared WITH the nn.MaxPool2d((1, 3)) that
                    aasist_model.py's copy omits. The SSL variant removed the
                    pooling because SSL feature maps are already short in time;
                    on a 64600-sample waveform the pooling is required, both to
                    reach the published receptive field and to make the node
                    count match pos_S.

Architecture and hyperparameters come from configs/AASIST.conf["model_config"]
(first_conv 128, filts [70, ...], gat_dims, pool_ratios, temperatures), i.e.
the published AASIST configuration already committed to this repo.

Tensor shapes for the benchmark's 64600-sample (~4 s) crop:
    (B, 64600) -> conv_time -> (B, 70, 64472) -> |.|, maxpool(3,3)
               -> (B, 1, 23, 21490) -> encoder (6x, time/3 each)
               -> (B, 64, 23, 29)   -> GAT-S over 23 spectral nodes
                                       GAT-T over 29 temporal nodes
               -> (B, 2) logits
"""

import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aasist_model import GraphAttentionLayer, GraphPool, HtrgGraphAttentionLayer

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "configs", "AASIST.conf")


class CONV(nn.Module):
    """Sinc band-pass convolution front-end (mel-spaced, fixed / non-learnable).

    Verbatim port of the published AASIST/RawNet2 CONV layer.
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 in_channels=1, stride=1, padding=0, dilation=1, bias=False,
                 groups=1, mask=False):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "SincConv only support one input channel "
                "(here, in_channels = {%i})" % in_channels)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetric)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * \
                np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * \
                np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        # Registered as a buffer (not a plain attribute as in the original) so
        # .to(device) moves it and it round-trips through state_dict. The
        # published code re-uploads it on every forward instead; a buffer is
        # equivalent and avoids a host->device copy per batch.
        self.register_buffer("band_pass", band_pass)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = int(np.random.uniform(0, 20))
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0

        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, filters, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, bias=None, groups=1)


class Residual_block(nn.Module):
    """AASIST residual block WITH temporal max-pooling.

    Distinct from aasist_model.Residual_block, which drops self.mp. See module
    docstring -- the pooling is required on raw-waveform input.
    """

    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


def load_model_config(path: Optional[str] = None) -> dict:
    """Read model_config out of configs/AASIST.conf."""
    with open(path or CONFIG_PATH, "r") as f:
        return json.loads(f.read())["model_config"]


class Model(nn.Module):
    """Standalone AASIST.

    Signature matches aasist_model.Model / linear_model.UtteranceLevel so
    main.py can instantiate it the same way: Model(args, device).
    """

    def __init__(self, args=None, device="cpu", d_args=None):
        super().__init__()
        self.device = device
        self.args = args

        d_args = d_args or load_model_config(
            getattr(args, "aasist_conf", None) if args is not None else None)
        self.d_args = d_args

        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        # Frequency-domain masking augmentation on the sinc filterbank. Off by
        # default: the published AASIST baseline does not use it, and the SSL
        # runs in this benchmark rely on RawBoost for augmentation instead.
        self.freq_aug = bool(getattr(args, "freq_aug", False)) if args is not None else False

        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=d_args["first_conv"],
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0],
                                               temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, Freq_aug=None):
        if Freq_aug is None:
            Freq_aug = self.freq_aug and self.training

        if x.dim() == 3:      # (B, T, 1) -> (B, T)
            x = x.squeeze(-1)
        x = x.unsqueeze(1)    # (B, 1, T)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # (#bs, #filt, #spec, #seq)
        e = self.encoder(x)

        # spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time
        e_S = e_S.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq
        e_T = e_T.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output
