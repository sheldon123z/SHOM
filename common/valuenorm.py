# -*- coding: utf-8 -*-
"""
@File      : valuenorm.py
@Time      : 2025-04-08 17:48
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 该文件实现了 `ValueNorm` 类，用于对观测向量进行归一化和反归一化操作。
- `ValueNorm` 类：继承自 `nn.Module`，用于对输入向量跨前 `norm_axes` 个维度进行归一化。
  - `__init__` 方法：初始化类的参数，包括输入形状、归一化轴数、衰减系数等，并初始化运行均值、运行平方均值和去偏项。
  - `running_mean_var` 方法：计算去偏的均值和方差。
  - `update` 方法：根据输入向量更新运行均值、运行平方均值和去偏项。
  - `normalize` 方法：对输入向量进行归一化处理。
  - `denormalize` 方法：将归一化后的数据转换回原始分布。

该类依赖 `numpy` 和 `torch` 库。
"""
"""ValueNorm."""
import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(
        self,
        input_shape,
        norm_axes=1,
        beta=0.99999,
        per_element_update=False,
        epsilon=1e-5,
        device=torch.device("cpu"),
    ):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape), requires_grad=False
        ).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(
            **self.tpdv
        )

    def running_mean_var(self):
        """Get running mean and variance."""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[: self.norm_axes])
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out

    def denormalize(self, input_vector):
        """Transform normalized data back into original distribution"""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        return out
