#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/06/2020
           """

import torch
from draugr.torch_utilities import conv2d_hw_shape, max_pool2d_hw_shape
from numpy import product
from torch import nn, relu

__all__ = ["AdversarialSignalClassifier"]


class AdversarialSignalClassifier(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
    ):
        super().__init__()

        self._in_shape = (in_channels, in_height, in_width)

        conv_stride = (1, 1)
        conv_kernel = (2, 2)
        padding = (0, 0)
        dilation = (1, 1)

        max_pool_1_kernel = (1, 3)
        max_pool_2_kernel = (1, 1)
        max_pool_3_kernel = (2, 2)

        self.convs = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.MaxPool2d(max_pool_1_kernel),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.MaxPool2d(max_pool_2_kernel),
            nn.ReLU(),
            nn.Conv2d(
                64,
                32,
                kernel_size=conv_kernel,
                stride=conv_stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.MaxPool2d(max_pool_3_kernel),
            nn.ReLU(),
        )

        self.num_flat = 32 * product(
            max_pool2d_hw_shape(
                conv2d_hw_shape(
                    max_pool2d_hw_shape(
                        conv2d_hw_shape(
                            max_pool2d_hw_shape(
                                conv2d_hw_shape(
                                    self._in_shape[1:], conv_kernel, conv_stride
                                ),
                                max_pool_1_kernel,
                            ),
                            conv_kernel,
                        ),
                        max_pool_2_kernel,
                    ),
                    conv_kernel,
                ),
                max_pool_3_kernel,
            )
        )

        self.fc1 = nn.Linear(self.num_flat, 128)
        self.fc2 = nn.Linear(128, 1)

        def init_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        self.convs.apply(init_weights)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")  # He
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._in_shape == x.shape[1:], f"x shape was {x.shape[1:]}"
        return self.fc2(relu(self.fc1(self.convs(x).reshape(-1, self.num_flat))))


if __name__ == "__main__":

    model = AdversarialSignalClassifier(3, 100, 100)
    print(model.__repr__())
