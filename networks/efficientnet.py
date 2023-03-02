import torch
from torch import nn, Tensor
from functools import partial
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union, Tuple
import collections
from itertools import repeat
import math

"""
Nd-EfficientNet based on  https://arxiv.org/abs/2104.00298
and
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
"""

def ConvNd(dim):
    return (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dim - 1]


def AdaptiveAvgPoolnD(dim):
    return (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)[dim - 1]


def ntuple(x, n):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class StochasticDepth(nn.Module):
    def __init__(self, p: float):
        """
        Deep Networks with Stochastic Depth
            https://arxiv.org/abs/1603.09382
        """
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        survival_rate = 1.0 - self.p
        noise = torch.empty([input.shape[0]] + [1] * (input.ndim - 1), dtype=input.dtype, device=input.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return input * noise

    def __repr__(self) -> str:
        return f"StochasticDepth(p={self.p})"


class SqueezeExcitation(nn.Module):
    """
    https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        dim,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = nn.Sigmoid,
    ):
        super().__init__()
        self.scale = nn.Sequential(
            AdaptiveAvgPoolnD(dim)(1),
            ConvNd(dim)(input_channels, squeeze_channels, 1),
            activation(),
            ConvNd(dim)(squeeze_channels, input_channels, 1),
            scale_activation(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale(x)


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        dim: int,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ):

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                kernel_size = ntuple(kernel_size, dim)
                dilation = ntuple(dilation, dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(dim))
        if bias is None:
            bias = norm_layer is None
        layers = [ConvNd(dim)(input_channels, output_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)]

        if norm_layer is not None:
            layers.append(norm_layer(output_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.output_channels = output_channels


class MBConv(nn.Module):
    def __init__(
        self,
        dim,
        fused,
        expand_ratio,
        kernel,
        stride,
        input_channels,
        output_channels,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        if stride == 1 and input_channels == output_channels:
            self.stochastic_depth = StochasticDepth(stochastic_depth_prob)
        else:
            self.stochastic_depth = None
        layers = []
        activation_layer = nn.SiLU
        expanded_channels = int((input_channels * expand_ratio) / 8) * 8

        if fused:
            if expanded_channels != input_channels:
                layers.append(ConvNormActivation(dim, input_channels, expanded_channels, kernel_size=kernel, stride=stride, norm_layer=norm_layer, activation_layer=activation_layer))
                layers.append(ConvNormActivation(dim, expanded_channels, output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None))
            else:
                layers.append(ConvNormActivation(dim, input_channels, output_channels, kernel_size=kernel, stride=stride, norm_layer=norm_layer, activation_layer=activation_layer))
        else:
            if expanded_channels != input_channels:
                layers.append(ConvNormActivation(dim, input_channels, expanded_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer))
            layers.append(ConvNormActivation(dim, expanded_channels, expanded_channels, kernel_size=kernel, stride=stride, groups=expanded_channels, norm_layer=norm_layer, activation_layer=activation_layer))
            squeeze_channels = max(1, input_channels // 4)
            layers.append(SqueezeExcitation(dim, expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))
            layers.append(ConvNormActivation(dim, expanded_channels, output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None))
        self.block = nn.Sequential(*layers)
        self.output_channels = output_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.stochastic_depth:
            return self.stochastic_depth(result) + input
        else:
            return result


@dataclass
class MBConfig:
    fused: bool
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    output_channels: int
    num_layers: int


class EfficientNet(nn.Module):
    def __init__(
        self,
        dim,
        input_channels: int,
        output_values: int,
        settings: Sequence[MBConfig] = (
            MBConfig(True, 1, 3, 1, 16, 16, 2),
            MBConfig(True, 4, 3, 2, 16, 32, 4),
            MBConfig(False, 4, 3, 2, 32, 64, 4),
            MBConfig(False, 4, 3, 2, 64, 128, 6),
        ),
        dropout: float = 0.3,
        stochastic_depth_prob: float = 0.2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ):
        """
        EfficientNetv2, https://arxiv.org/abs/2104.00298
        Args:
            dim: dimension (2,3)
            input_channels: number of input channels
            output_values: number of output values
            settings: Network structure
            dropout: droupout probability
            stochastic_depth_prob: stochastic depth probability
            num_classes (int): Number of classes
            norm_layer: Normalization layer to use, default GroupNorm(num_groups=8)
            last_channel: number of channels on the penultimate layer
        """
        super().__init__()

        if norm_layer is None:
            norm_layer = lambda channels: nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-3)
        layers = [ConvNormActivation(dim, input_channels, settings[0].input_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU)]
        total_blocks = sum(config.num_layers for config in settings)
        block_counter = 0.0
        for config in settings:
            stage = []
            for _ in range(config.num_layers):
                sd_prob = stochastic_depth_prob * block_counter / total_blocks
                stage.append(
                    MBConv(
                        dim,
                        config.fused,
                        config.expand_ratio,
                        config.kernel,
                        1 if stage else config.stride,
                        config.output_channels if stage else config.input_channels,
                        config.output_channels,
                        sd_prob,
                        norm_layer,
                    )
                )
                block_counter += 1
            layers.append(nn.Sequential(*stage))
        if last_channel is None:
            last_channel = 4 *  settings[-1].output_channels
        layers.append(ConvNormActivation(dim, settings[-1].output_channels, last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU))

        self.features = nn.Sequential(*layers)
        self.avgpool = AdaptiveAvgPoolnD(dim)(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, output_values),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
