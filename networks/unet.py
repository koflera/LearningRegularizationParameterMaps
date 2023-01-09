import torch
import torch.nn as nn

import numpy as np


class ConvBlock(nn.Module):

    """
    A block of convolutional layers (1D, 2D or 3D)
    """

    def __init__(self, dim, n_ch_in, n_ch_out, n_convs, kernel_size=3, bias=False, padding_mode="zeros"):
        super().__init__()

        if dim == 1:
            conv_op = nn.Conv1d
        if dim == 2:
            conv_op = nn.Conv2d
        elif dim == 3:
            conv_op = nn.Conv3d

        padding = np.int(np.floor(kernel_size / 2))

        conv_block_list = []
        conv_block_list.extend([conv_op(n_ch_in, n_ch_out, kernel_size, padding=padding, bias=bias, padding_mode=padding_mode), nn.LeakyReLU()])

        for i in range(n_convs - 1):
            conv_block_list.extend([conv_op(n_ch_out, n_ch_out, kernel_size, padding=padding, bias=bias, padding_mode=padding_mode), nn.LeakyReLU()])

        self.conv_block = nn.Sequential(*conv_block_list)

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, dim, n_ch_in, n_enc_stages, n_convs_per_stage, n_filters, kernel_size=3, bias=False, padding_mode="zeros"):
        super().__init__()

        n_ch_list = [n_ch_in]
        for ne in range(n_enc_stages):
            n_ch_list.append(np.int(n_filters) * 2 ** ne)

        self.enc_blocks = nn.ModuleList(
            [ConvBlock(dim, n_ch_list[i], n_ch_list[i + 1], n_convs_per_stage, kernel_size=kernel_size, bias=bias, padding_mode=padding_mode) for i in range(len(n_ch_list) - 1)]
        )

        if dim == 1:
            pool_op = nn.MaxPool1d(2)
        elif dim == 2:
            pool_op = nn.MaxPool2d(2)
        elif dim == 3:
            pool_op = nn.MaxPool3d(2)

        self.pool = pool_op

    def forward(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, dim, n_ch_in, n_dec_stages, n_convs_per_stage, n_filters, kernel_size=3, bias=False, padding_mode="zeros"):
        super().__init__()

        n_ch_list = []
        for ne in range(n_dec_stages):
            n_ch_list.append(np.int(n_ch_in * (1 / 2) ** ne))

        if dim == 1:
            conv_op = nn.Conv1d
            interp_mode = "linear"
        elif dim == 2:
            conv_op = nn.Conv2d
            interp_mode = "bilinear"
        elif dim == 3:
            interp_mode = "trilinear"
            conv_op = nn.Conv3d

        self.interp_mode = interp_mode

        padding = np.int(np.floor(kernel_size / 2))
        self.upconvs = nn.ModuleList([conv_op(n_ch_list[i], n_ch_list[i + 1], kernel_size=kernel_size, padding=padding, bias=bias, padding_mode=padding_mode) for i in range(len(n_ch_list) - 1)])
        self.dec_blocks = nn.ModuleList(
            [ConvBlock(dim, n_ch_list[i], n_ch_list[i + 1], n_convs_per_stage, kernel_size=kernel_size, bias=bias, padding_mode=padding_mode) for i in range(len(n_ch_list) - 1)]
        )

    def forward(self, x, encoder_features):

        for i in range(len(self.dec_blocks)):
            # x        = self.upconvs[i](x)
            enc_features = encoder_features[i]
            enc_features_shape = enc_features.shape
            x = nn.functional.interpolate(x, enc_features_shape[2:], mode=self.interp_mode, align_corners=False)
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, dim, n_ch_in=2, n_ch_out=2, n_enc_stages=3, n_convs_per_stage=2, n_filters=16, kernel_size=3, res_connection=False, bias=True, padding_mode="zeros"):
        super().__init__()
        self.encoder = Encoder(dim, n_ch_in, n_enc_stages, n_convs_per_stage, n_filters, kernel_size=kernel_size, bias=bias, padding_mode=padding_mode)
        self.decoder = Decoder(
            dim, n_filters * (2 ** (n_enc_stages - 1)), n_enc_stages, n_convs_per_stage, n_filters * (n_enc_stages * 2), kernel_size=kernel_size, bias=bias, padding_mode=padding_mode,
        )

        if dim == 1:
            conv_op = nn.Conv1d
        elif dim == 2:
            conv_op = nn.Conv2d
        elif dim == 3:
            conv_op = nn.Conv3d

        self.c1x1 = conv_op(n_filters, n_ch_out, kernel_size=1, padding=0, bias=bias)
        if res_connection:
            if n_ch_in == n_ch_out:
                self.res_connection = lambda x: x
            else:
                self.res_connection = conv_op(n_ch_in, n_ch_out, 1)
        else:
            self.res_connection = False

    def forward(self, x):
        enc_features = self.encoder(x)
        dec = self.decoder(enc_features[-1], enc_features[::-1][1:])
        out = self.c1x1(dec)
        if self.res_connection:
            out = out + self.res_connection(x)
        return out
