import copy
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
from matplotlib import pyplot as plt

plt.switch_backend("agg")
import numpy as np
import torch
from torch import nn, optim
from torch.utils import tensorboard as tb
from tqdm import tqdm
from typing import *


class ReplayBuffer:
    def __init__(self, capicity: int) -> None:
        self.capicity = capicity
        self.buffer = deque(maxlen=self.capicity)

    @property
    def size(self):
        return len(self.buffer)

    def push(self, s, a, r, next_s, t):
        if self.size == self.capicity:
            self.buffer.popleft()
        self.buffer.append([s, a, r, next_s, t])

    def is_full(self):
        return self.size == self.capicity

    def sample(self, N: int, device: str):
        """采样数据并打包"""
        assert N <= self.size, "batch is too big"
        samples = random.sample(self.buffer, N)
        states, actions, rewards, next_states, terminated = zip(*samples)
        return (
            torch.from_numpy(np.vstack(states)).float().to(device),
            torch.from_numpy(np.vstack(actions)).float().to(device),
            torch.from_numpy(np.vstack(rewards)).float().to(device),
            torch.from_numpy(np.vstack(next_states)).float().to(device),
            torch.from_numpy(np.vstack(terminated)).float().to(device),
        )


class Actor(nn.Module):
    """actor网络"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound: np.ndarray):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        # tanh将输出限制在(-1,+1)之间
        self.tanh = nn.Tanh()
        # action_bound是环境可以接受的动作最大值
        self.action_bound = torch.from_numpy(action_bound).float()

    def forward(self, state_tensor):
        x = self.fc1(state_tensor)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x * self.action_bound


class Critic(nn.Module):
    """Q网络: (s,a)-->q"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state_tensor, action_tensor):
        """网络输入是状态和动作, 因此需要cat在一起"""
        x = torch.cat([state_tensor, action_tensor], dim=-1)  # 拼接状态和动作
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class DDPG:
    def __init__(
        self,
        env: Union[gym.Env, str],
        heter: float = 0.5,
        lr: float = 1e-3,
        sigma: float = 0.10,
        tau: float = 0.005,
        gamma: float = 0.98,
        hidden_dim: int = 400,
        buffer_capicity: int = 10000,
        buffer_init_ratio: float = 0.30,
        batch_size: int = 64,
        device: str = "cpu",
        train_batchs: int = None,
        save_dir: str = None,
        **kwargs,
    ):
        if isinstance(env, str):
            self.env = gym.make(env, heter=heter)
            self.env_name = env
        else:
            self.env = env
            self.env_name = self.env.spec.id

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_bound = self.env.action_space.high
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr / 10)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr)
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.replay_buffer = ReplayBuffer(buffer_capicity)
        self.batch_size = batch_size

        # 训练时使用
        self.save_dir = Path(save_dir)
        self.episode = 0
        self.episode_reward = 0
        self.episode_reward_list = []
        self.episode_len = 0
        self.global_step = 0
        self.total_train_batchs = train_batchs
        self.logger = None
        self.save_dir.mkdir(parents=True)
        self.logger = tb.SummaryWriter(self.save_dir / "log")
        self.collect_exp_before_train(buffer_init_ratio)
        # 开始训练
        self.state, _ = self.env.reset()

    @torch.no_grad()
    def get_action(self, s, add_noise=True):
        """在训练时得到含噪声的连续动作"""
        if isinstance(s, np.ndarray) is True:
            s = torch.from_numpy(s).float().to(self.device)
        a = self.actor(s)
        if add_noise:
            a += torch.normal(0.0, self.sigma, a.shape)
        return a.cpu().numpy()

    def collect_exp_before_train(self, ratio: float):
        """开启训练之前预先往buffer里面存入一定数量的经验"""
        assert 0 < ratio < 1
        num = ratio * self.replay_buffer.capicity
        s, _ = self.env.reset()
        while self.replay_buffer.size < num:
            a = self.env.action_space.sample()
            ns, r, t1, _, _ = self.env.step(a)
            self.replay_buffer.push(s, a, r, ns, t1)
            s = ns if not t1 else self.env.reset()[0]

    def soft_sync_target(self):
        """定期同步权重参数到target"""
        net_groups = [(self.actor, self.actor_target), (self.critic, self.critic_target)]
        for net, net_ in net_groups:
            for p, p_ in zip(net.parameters(), net_.parameters()):
                p_.data.copy_(p.data * self.tau + p_.data * (1 - self.tau))

    def train_one_batch(self):
        # 从buffer中取出一批数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)
        # 计算critic_loss并更新
        with torch.no_grad():
            td_targets = rewards + self.gamma * (1 - dones) * self.critic_target(
                next_states, self.actor_target(next_states)
            )
        td_errors = td_targets - self.critic(states, actions)
        critic_loss = torch.pow(td_errors, 2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 计算actor_loss并更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 软更新target
        self.soft_sync_target()
        return actor_loss.detach(), critic_loss.detach()

    def train(self, batch_nums: int):
        # 训练前准备
        for _ in range(batch_nums):
            a = self.get_action(self.state)
            ns, r, t1, t2, _ = self.env.step(a)
            self.episode_reward += r
            self.replay_buffer.push(self.state, a, r, ns, t1)
            if t1 or t2:
                self.log_info_per_episode()
                self.state, _ = self.env.reset()
            else:
                self.state = ns
            actor_loss, critic_loss = self.train_one_batch()
            self.log_info_per_batch(actor_loss, critic_loss)

    def log_info_per_episode(self):
        self.logger.add_scalar("Train/episode_reward", self.episode_reward, self.episode)
        self.logger.add_scalar("Train/buffer_size", self.replay_buffer.size, self.episode)
        self.logger.add_scalar("Episode/episode_len", self.episode_len, self.episode)
        self.episode_reward_list.append(self.episode_reward)
        self.episode += 1
        self.episode_len = 0
        self.episode_reward = 0

    def log_info_per_batch(self, actor_loss, critic_loss):
        self.logger.add_scalar("Loss/actor_loss", actor_loss, self.global_step)
        self.logger.add_scalar("Loss/critic_loss", critic_loss, self.global_step)
        self.global_step += 1
        self.episode_len += 1


class Server:
    """server角色"""

    def __init__(self, points: List[DDPG], device: str = "cpu") -> None:
        """为保护用户隐私, 除了神经网络参数之外, 不能从节点读取任何数据"""
        self.points = points
        self.device = device
        self.actor = copy.deepcopy(self.points[0].actor).to(self.device)
        self.actor_target = copy.deepcopy(self.points[0].actor_target).to(self.device)
        self.critic = copy.deepcopy(self.points[0].critic).to(self.device)
        self.critic_target = copy.deepcopy(self.points[0].critic_target).to(self.device)

    def merge_params(self, merge_target: bool = False) -> None:
        """合并/分发参数"""
        for name, param in self.actor.state_dict().items():
            avg_param = torch.stack([p.actor.state_dict()[name] for p in self.points]).mean(dim=0)
            param.data.copy_(avg_param.data)
        for name, param in self.critic.state_dict().items():
            avg_param = torch.stack([p.critic.state_dict()[name] for p in self.points]).mean(dim=0)
            param.data.copy_(avg_param.data)
        if merge_target is True:
            for name, param in self.actor_target.state_dict().items():
                avg_param = torch.stack([p.actor_target.state_dict()[name] for p in self.points]).mean(dim=0)
                param.data.copy_(avg_param.data)
            for name, param in self.critic_target.state_dict().items():
                avg_param = torch.stack([p.critic_target.state_dict()[name] for p in self.points]).mean(dim=0)
                param.data.copy_(avg_param.data)
        for p in self.points:
            p.actor.load_state_dict(self.actor.state_dict())
            p.critic.load_state_dict(self.critic.state_dict())
            if merge_target is True:
                p.actor_target.load_state_dict(self.actor_target.state_dict())
                p.critic_target.load_state_dict(self.critic_target.state_dict())


class FedDDPG:
    def __init__(
        self,
        point_configs: List[dict],
        merge_num: int,
        merge_interval: int,
        merge_target: bool,
        episode_num_eval: int,
        save_dir: str = None,
        device: str = "cpu",
    ) -> None:
        assert save_dir is not None, "save_dir can't be empty"
        self.device = device
        self.point_configs = point_configs
        self.merge_num = merge_num
        self.merge_interval = merge_interval
        self.merge_target = merge_target
        self.episode_num_eval = episode_num_eval
        self.save_dir = save_dir

        self.points = [DDPG(**c) for c in point_configs]
        self.server = Server(self.points, device=self.device)
        self.logger = tb.SummaryWriter(self.save_dir / "global" / "log")

    def train(self):
        """总共合并训练self.merge_num次"""
        bar = tqdm(range(self.merge_num))
        for n in bar:
            for p in tqdm(self.points, leave=False, disable=True):
                p.train(self.merge_interval)
            self.server.merge_params(self.merge_target)
            avg_merge_episode_reward = self.evaluate_avg_reward()
            bar.set_description_str(f"reward->{int(avg_merge_episode_reward):3d}|")
            self.logger.add_scalar("aggregate/reward", avg_merge_episode_reward, global_step=n)
            self.save(self.save_dir / "server" / f"aggre_{n}.pt")
        self.summarize_point_reward()
        for p in self.points:
            p.logger.close()
        self.logger.close()

    def train_baseline(self):
        """训练baseline用于对照"""
        batch_nums = self.points[0].total_train_batchs
        for p in tqdm(self.points, desc="Training baseline..."):
            p.train(batch_nums)
        self.summarize_point_reward()
        env_num = len(self.points)
        table = np.zeros((env_num, env_num))
        for i in tqdm(range(env_num), desc="Evaluating..."):
            for j in range(env_num):
                point_r = 0
                s, _ = self.points[j].env.reset()
                while True:
                    a = self.points[i].get_action(s, add_noise=False)
                    next_s, r, t1, t2, _ = self.points[j].env.step(a)
                    point_r += r
                    s = next_s
                    if t1 or t2:
                        break
                table[i][j] = point_r
        np.save(self.save_dir / "baseline.npy", table)
        for p in self.points:
            p.logger.close()
        self.logger.close()
        return table.mean()

    def evaluate_avg_reward(self):
        """每次merge之后每个节点评估一下奖励"""
        reward_list = []
        for p in self.points:
            env = copy.deepcopy(p.env)
            point_r = 0
            for _ in range(self.episode_num_eval):
                s, _ = env.reset()
                while True:
                    a = p.get_action(s, add_noise=False)
                    ns, r, t1, t2, _ = env.step(a)
                    point_r += r
                    s = ns
                    if t1 or t2:
                        break
            reward_list.append(point_r / self.episode_num_eval)
        return sum(reward_list) / len(reward_list)

    def summarize_point_reward(self):
        """统计每个point已经完成的episode的奖励, 并按最短的长度取平均"""
        min_length = min([len(p.episode_reward_list) for p in self.points])
        table = []
        for p in self.points:
            table.append(p.episode_reward_list[:min_length])
            np.save(p.save_dir / "episode_reward_list.npy", np.array(p.episode_reward_list))
        avg_episode_reward = np.array(table).mean(0)
        plt.plot(range(min_length), avg_episode_reward), plt.grid(), plt.title("average episode reward")
        plt.savefig(self.save_dir / "global" / "average_episode_reward.svg")
        plt.close()

    def save(self, save_path):
        """保存权重"""
        Path(save_path).parent.mkdir(exist_ok=True)
        params = {
            "weights": [
                {"actor": self.server.actor.state_dict()},
                {"critic": self.server.critic.state_dict()},
            ]
        }
        # for p in self.points:
        #     if p.embedding is not None:
        #         params.update({f"embedding_{p.env_index}": p.embedding})
        torch.save(params, save_path)
