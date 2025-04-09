# -*- coding: utf-8 -*-
"""
@File      : haddpg.py
@Time      : 2025-04-08 17:45
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 该文件实现了HADDPG（Hierarchical Actor - Critic Deep Deterministic Policy Gradient）算法，属于离线策略算法。
- 关键组件及职责：
  - HADDPG类：继承自OffPolicyBase，用于实现HADDPG算法的核心逻辑。
    - __init__方法：初始化HADDPG算法所需的参数、策略网络、目标网络和优化器。
    - get_actions方法：根据观测值获取动作，可选择是否添加噪声。
    - get_target_actions方法：根据观测值获取目标策略网络的动作。
- 依赖模块：
  - deepcopy：用于复制对象。
  - torch：用于构建和训练神经网络。
  - DeterministicPolicy：用于定义确定性策略网络。
  - check：用于检查输入数据。
  - OffPolicyBase：离线策略算法的基类。
- 输入输出：
  - 输入：观测值、是否添加噪声的标志。
  - 输出：动作张量。
"""
"""HADDPG algorithm."""
from copy import deepcopy
import torch
from models.policy_models.deterministic_policy import DeterministicPolicy
from utils.envs_tools import check
from algorithms.actors.off_policy_base import OffPolicyBase


class HADDPG(OffPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        assert (
            act_space.__class__.__name__ == "Box"
        ), f"only continuous action space is supported by {self.__class__.__name__}."
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]

        self.actor = DeterministicPolicy(args, obs_space, act_space, device)
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.low = torch.tensor(act_space.low).to(**self.tpdv)
        self.high = torch.tensor(act_space.high).to(**self.tpdv)
        self.scale = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

    def get_actions(self, obs, add_noise):
        """Get actions for observations.
        Args:
            obs: (np.ndarray) observations of actor, shape is (n_threads, dim) or (batch_size, dim)
            add_noise: (bool) whether to add noise
        Returns:
            actions: (torch.Tensor) actions taken by this actor, shape is (n_threads, dim) or (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs)
        if add_noise:
            actions += torch.randn_like(actions) * self.expl_noise * self.scale
            actions = torch.clamp(actions, self.low, self.high)
        return actions

    def get_target_actions(self, obs):
        """Get target actor actions for observations.
        Args:
            obs: (np.ndarray) observations of target actor, shape is (batch_size, dim)
        Returns:
            actions: (torch.Tensor) actions taken by target actor, shape is (batch_size, dim)
        """
        obs = check(obs).to(**self.tpdv)
        return self.target_actor(obs)
