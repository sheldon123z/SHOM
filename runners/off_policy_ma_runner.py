# -*- coding: utf-8 -*-
"""
@File      : off_policy_ma_runner.py
@Time      : 2025-04-08 17:41
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 该文件是一个用于离线策略多智能体（MA）算法的运行器脚本，借助 PyTorch 实现。
- `OffPolicyMARunner` 类：继承自 `OffPolicyBaseRunner`，负责运行离线策略多智能体算法。
  - `train` 方法：
    - 从缓冲区采样数据，包括共享观测、智能体观测、动作等多种数据。
    - 训练评论家网络（critic）：先开启梯度，计算下一动作，调用 `critic.train` 方法训练，再关闭梯度。
    - 按指定频率训练行动者网络（actor）：开启梯度，计算动作，获取价值预测，计算损失并更新参数，最后关闭梯度。
    - 执行软更新：更新行动者和评论家网络的目标网络参数。
"""
"""Runner for off-policy MA algorithms"""
import copy
import torch
from runners.off_policy_base_runner import OffPolicyBaseRunner


class OffPolicyMARunner(OffPolicyBaseRunner):
    """Runner for off-policy MA algorithms."""

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        next_actions = []
        for agent_id in range(self.num_agents):
            next_actions.append(
                self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
            )
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_term,
            sp_next_share_obs,
            next_actions,
            sp_gamma,
        )
        self.critic.turn_off_grad()
        if self.total_it % self.policy_freq == 0:
            # train actors
            # actions shape: (n_agents, batch_size, dim)
            for agent_id in range(self.num_agents):
                actions = copy.deepcopy(torch.tensor(sp_actions)).to(self.device)
                self.actor[agent_id].turn_on_grad()
                # train this agent
                actions[agent_id] = self.actor[agent_id].get_actions(
                    sp_obs[agent_id], False
                )
                actions_list = [a for a in actions]
                actions_t = torch.cat(actions_list, dim=-1)
                value_pred = self.critic.get_values(sp_share_obs, actions_t)
                actor_loss = -torch.mean(value_pred)
                self.actor[agent_id].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[agent_id].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()
            # soft update
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
            self.critic.soft_update()
