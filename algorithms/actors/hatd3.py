# -*- coding: utf-8 -*-
"""
@File      : hatd3.py
@Time      : 2025-04-08 17:45
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此 Python 文件实现了 HATD3 算法，继承自 HADDPG 算法。
- 关键类：
  - HATD3 类：继承 HADDPG 类，实现 HATD3 算法核心逻辑。
    - 初始化时，接收参数、观测空间、动作空间和设备，还初始化策略噪声和噪声剪辑参数。
    - get_target_actions 函数：根据观测值获取目标动作。
      - 输入：观测值数组（形状为 (batch_size, dim)）。
      - 处理：将观测值转换为张量，通过目标演员网络得到动作，添加噪声并剪辑，最后将动作剪辑到合法范围。
      - 输出：目标演员采取的动作张量（形状为 (batch_size, dim)）。
- 依赖库：
  - torch：用于张量计算和神经网络操作。
  - utils.envs_tools 中的 check 函数：用于数据检查。
  - algorithms.actors.haddpg 中的 HADDPG 类：作为 HATD3 类的基类。
"""
"""HATD3 algorithm."""
import torch
from utils.envs_tools import check
from algorithms.actors.haddpg import HADDPG


class HATD3(HADDPG):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)
        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.target_actor(obs)
        noise = torch.randn_like(actions) * self.policy_noise * self.scale
        noise = torch.clamp(
            noise, -self.noise_clip * self.scale, self.noise_clip * self.scale
        )
        actions += noise
        actions = torch.clamp(actions, self.low, self.high)
        return actions
