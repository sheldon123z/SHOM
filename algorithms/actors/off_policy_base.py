# -*- coding: utf-8 -*-
"""
@File      : off_policy_base.py
@Time      : 2025-04-08 17:45
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此 Python 文件定义了一个用于离策略（off - policy）算法的基类 `OffPolicyBase`，主要作用是为离策略算法提供基础功能和通用接口。
- 关键组件及职责：
  - `__init__`：类的构造函数，初始化对象，接收参数、观测空间、动作空间和设备信息。
  - `lr_decay`：衰减 actor 和 critic 的学习率，根据当前训练步骤和总训练步骤更新学习率。
  - `get_actions`：获取动作，具体实现待子类完成。
  - `get_target_actions`：获取目标动作，具体实现待子类完成。
  - `soft_update`：软更新目标 actor 的参数，使用 Polyak 平均法更新。
  - `save`：保存 actor 和目标 actor 的状态字典到指定目录。
  - `restore`：从指定目录恢复 actor 和目标 actor 的状态字典。
  - `turn_on_grad`：开启 actor 参数的梯度计算。
  - `turn_off_grad`：关闭 actor 参数的梯度计算。
- 依赖库：使用了 `copy`、`numpy`、`torch` 库，以及 `utils.envs_tools` 和 `utils.models_tools` 模块。
"""
"""Base class for off-policy algorithms."""

from copy import deepcopy
import numpy as np
import torch
from utils.envs_tools import check
from utils.models_tools import update_linear_schedule


class OffPolicyBase:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        pass

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.actor_optimizer, step, steps, self.lr)

    def get_actions(self, obs, randomness):
        pass

    def get_target_actions(self, obs):
        pass

    def soft_update(self):
        """Soft update target actor."""
        for param_target, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def save(self, save_dir, id):
        """Save the actor and target actor."""
        torch.save(
            self.actor.state_dict(), str(save_dir) + "/actor_agent" + str(id) + ".pt"
        )
        torch.save(
            self.target_actor.state_dict(),
            str(save_dir) + "/target_actor_agent" + str(id) + ".pt",
        )

    def restore(self, model_dir, id):
        """Restore the actor and target actor."""
        actor_state_dict = torch.load(str(model_dir) + "/actor_agent" + str(id) + ".pt")
        self.actor.load_state_dict(actor_state_dict)
        target_actor_state_dict = torch.load(
            str(model_dir) + "/target_actor_agent" + str(id) + ".pt"
        )
        self.target_actor.load_state_dict(target_actor_state_dict)

    def turn_on_grad(self):
        """Turn on grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """Turn off grad for actor parameters."""
        for p in self.actor.parameters():
            p.requires_grad = False
