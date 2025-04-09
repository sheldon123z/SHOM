# -*- coding: utf-8 -*-
"""
@File      : continuous_q_critic.py
@Time      : 2025-04-08 17:45
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此 Python 文件实现了一个连续 Q 评论家（Continuous Q Critic），用于学习连续动作空间中的 Q 函数。
- 关键组件及职责：
  - `ContinuousQCritic` 类：核心类，负责初始化评论家及相关参数，管理训练过程。
    - `__init__` 方法：初始化评论家，设置设备、动作空间、优化器等参数。
    - `lr_decay` 方法：衰减评论家的学习率。
    - `soft_update` 方法：软更新目标网络。
    - `get_values` 方法：获取 Q 值。
    - `train` 方法：训练评论家。
    - `save` 方法：保存模型。
    - `restore` 方法：恢复模型。
    - `turn_on_grad` 方法：开启评论家的梯度计算。
    - `turn_off_grad` 方法：关闭评论家的梯度计算。
- 依赖库：
  - `torch`：用于深度学习计算。
  - `deepcopy`：用于复制模型。
  - `ContinuousQNet`：连续 Q 网络模型。
  - `check`：环境工具函数。
  - `update_linear_schedule`：线性学习率衰减工具函数。
"""
"""Continuous Q Critic."""
from copy import deepcopy
import torch
from models.value_function_models.continuous_q_net import ContinuousQNet
from utils.envs_tools import check
from utils.models_tools import update_linear_schedule


class ContinuousQCritic:
    """Continuous Q Critic.
    Critic that learns a Q-function. The action space is continuous.
    Note that the name ContinuousQCritic emphasizes its structure that takes observations and actions as input and
    outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space. For now, it only supports continuous action space, but we will enhance its capability to
    include discrete action space in the future.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.critic_optimizer, step, steps, self.critic_lr)

    def soft_update(self):
        """Soft update the target network."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def get_values(self, share_obs, actions):
        """Get the Q values."""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return self.critic(share_obs, actions)

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """Train the critic.
        Args:
            share_obs: (np.ndarray) shape is (batch_size, dim)
            actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            reward: (np.ndarray) shape is (batch_size, 1)
            done: (np.ndarray) shape is (batch_size, 1)
            term: (np.ndarray) shape is (batch_size, 1)
            next_share_obs: (np.ndarray) shape is (batch_size, dim)
            next_actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            gamma: (np.ndarray) shape is (batch_size, 1)
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(**self.tpdv)
        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_q_values = self.target_critic(next_share_obs, next_actions)
        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)
        critic_loss = torch.mean(
            torch.nn.functional.mse_loss(self.critic(share_obs, actions), q_targets)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, save_dir):
        """Save the model."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )

    def restore(self, model_dir):
        """Restore the model."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)

    def turn_on_grad(self):
        """Turn on the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic."""
        for param in self.critic.parameters():
            param.requires_grad = False
