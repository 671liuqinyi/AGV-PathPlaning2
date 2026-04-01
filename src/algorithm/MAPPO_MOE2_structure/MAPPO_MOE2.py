import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithm.MAPPO_MOE2_structure.features import EXPERT_FEATURE_DIM


class MoE2PerTypeActorNet(nn.Module):
    """
    MoE actor with 3 expert types and 2 sub-experts for each type (total 6 experts):
    - Type 0: straight dash
      - sub 0: long-distance dash
      - sub 1: short-distance fine move
    - Type 1: obstacle avoidance
      - sub 0: static obstacle bypass
      - sub 1: narrow-corridor pass
    - Type 2: yielding/game-theoretic avoidance
      - sub 0: conservative yielding
      - sub 1: active side-step yielding
    """

    def __init__(self, num_inputs=3, num_actions=4, expert_feature_dim=EXPERT_FEATURE_DIM):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_types = 3
        self.sub_per_type = 2
        self.num_experts = self.num_types * self.sub_per_type
        self.expert_feature_dim = int(expert_feature_dim)

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc_shared = nn.Linear(64 * 4 * 4, 128)
        self.expert_heads = nn.ModuleList([nn.Linear(128, self.num_actions) for _ in range(self.num_experts)])

        gate_in_dim = 128 + self.expert_feature_dim
        self.type_gate_head = nn.Linear(gate_in_dim, self.num_types)
        self.sub_gate_heads = nn.ModuleList([nn.Linear(gate_in_dim, self.sub_per_type) for _ in range(self.num_types)])

    def _encode_obs(self, obs_tensor):
        x = F.relu(self.conv1(obs_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc_shared(x))

    def _heuristic_type_bias(self, feature_tensor):
        dx = feature_tensor[:, 0]
        dy = feature_tensor[:, 1]
        dist = feature_tensor[:, 2]
        free = feature_tensor[:, 3:7]
        occ = feature_tensor[:, 7:11]
        near_dist = feature_tensor[:, 13]
        conflict = feature_tensor[:, 14:18]
        open_ratio = feature_tensor[:, 18]

        blocked_ratio = 1.0 - open_ratio
        near_inv = 1.0 - near_dist
        adj_occ = occ.mean(dim=1)
        conflict_mean = conflict.mean(dim=1)

        straight_score = 1.4 * open_ratio + 1.2 * dist - 0.7 * near_inv - 0.8 * adj_occ
        obstacle_score = 1.8 * blocked_ratio + 0.7 * conflict_mean
        yielding_score = 1.6 * adj_occ + 1.8 * conflict_mean + 1.2 * near_inv

        base_straight = torch.stack([-dy, dx, dy, -dx], dim=1) * 2.4 + (free * 2.0 - 1.0) * 0.4
        base_obstacle = free * 2.2 - (1.0 - free) * 2.5 + base_straight * 0.2
        away = torch.stack([feature_tensor[:, 12], -feature_tensor[:, 11], -feature_tensor[:, 12], feature_tensor[:, 11]], dim=1)
        base_yielding = free * 1.2 - conflict * 3.1 - occ * 2.1 + away * 1.5
        base_bias = torch.stack([base_straight, base_obstacle, base_yielding], dim=1)  # [B,3,A]

        type_score = torch.stack([straight_score, obstacle_score, yielding_score], dim=1)
        return base_bias, type_score

    def _heuristic_subtype_bias(self, feature_tensor):
        dist = feature_tensor[:, 2]
        open_ratio = feature_tensor[:, 18]
        blocked_ratio = 1.0 - open_ratio
        near_dist = feature_tensor[:, 13]
        conflict = feature_tensor[:, 14:18].mean(dim=1)

        # type 0: long vs short distance
        t0 = torch.stack([1.6 * dist, 1.6 * (1.0 - dist)], dim=1)
        # type 1: static bypass vs corridor pass
        t1 = torch.stack([1.4 * blocked_ratio, 1.1 * open_ratio + 0.3 * blocked_ratio], dim=1)
        # type 2: conservative yield vs active side-step
        t2 = torch.stack([1.4 * conflict + 1.0 * (1.0 - near_dist), 1.2 * conflict + 0.6 * near_dist], dim=1)
        return torch.stack([t0, t1, t2], dim=1)  # [B,3,2]

    def forward(self, obs_tensor, feature_tensor, action_mask=None):
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        feature_tensor = feature_tensor.float()

        shared = self._encode_obs(obs_tensor)
        gate_input = torch.cat([shared, feature_tensor], dim=1)

        base_bias, type_score_prior = self._heuristic_type_bias(feature_tensor)
        sub_score_prior = self._heuristic_subtype_bias(feature_tensor)

        # [B,3]
        type_logits = self.type_gate_head(gate_input) + type_score_prior
        type_weights = F.softmax(type_logits, dim=1)

        sub_weights_list = []
        for t in range(self.num_types):
            sub_logits = self.sub_gate_heads[t](gate_input) + sub_score_prior[:, t, :]
            sub_weights_list.append(F.softmax(sub_logits, dim=1))
        # [B,3,2]
        sub_weights = torch.stack(sub_weights_list, dim=1)

        # [B,3,2,A]
        expert_logits = []
        for t in range(self.num_types):
            for s in range(self.sub_per_type):
                idx = t * self.sub_per_type + s
                one_logit = self.expert_heads[idx](shared) + base_bias[:, t, :]
                expert_logits.append(one_logit)
        expert_logits = torch.stack(expert_logits, dim=1).view(-1, self.num_types, self.sub_per_type, self.num_actions)

        expert_weights = type_weights.unsqueeze(-1) * sub_weights  # [B,3,2]
        mixed_logits = torch.sum(expert_weights.unsqueeze(-1) * expert_logits, dim=(1, 2))  # [B,A]

        if action_mask is not None:
            mask = action_mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            valid_any = mask.any(dim=1, keepdim=True)
            safe_mask = torch.where(valid_any, mask, torch.ones_like(mask, dtype=torch.bool))
            mixed_logits = mixed_logits.masked_fill(~safe_mask, -1e9)

        return mixed_logits, type_weights, sub_weights


class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.global_states = []
        self.action_masks = []
        self.expert_features = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.actions)


class MAPPOMoE2Agent:
    def __init__(self, actor_net, critic_net, device):
        self.device = device
        self.actor_net = actor_net.to(device)
        self.critic_net = critic_net.to(device)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.update_epochs = 3
        self.minibatch_size = 256
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.type_balance_coef = 0.02
        self.sub_balance_coef = 0.01
        self.max_grad_norm = 0.5

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=1e-3)

        self.buffer = RolloutBuffer()
        self.policy_loss_value = []
        self.value_loss_value = []
        self.type_balance_value = []
        self.sub_balance_value = []

    def choose_action(self, obs, expert_features, action_mask=None, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature_tensor = torch.tensor(expert_features, dtype=torch.float32, device=self.device).unsqueeze(0)

        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        logits, type_weights, sub_weights = self.actor_net(obs_tensor, feature_tensor, mask_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        type_w = type_weights[0].detach().cpu().numpy()
        sub_w = sub_weights[0].detach().cpu().numpy().reshape(-1)
        return int(action.item()), float(log_prob.item()), type_w, sub_w

    def evaluate_value(self, global_state):
        state_tensor = torch.tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.critic_net(state_tensor)
        return float(value.item())

    def store_transition(self, obs, global_state, action_mask, expert_features, action, log_prob, value, reward, done):
        self.buffer.obs.append(np.array(obs, dtype=np.float32))
        self.buffer.global_states.append(np.array(global_state, dtype=np.float32))
        self.buffer.action_masks.append(np.array(action_mask, dtype=np.float32))
        self.buffer.expert_features.append(np.array(expert_features, dtype=np.float32))
        self.buffer.actions.append(int(action))
        self.buffer.log_probs.append(float(log_prob))
        self.buffer.values.append(float(value))
        self.buffer.rewards.append(float(reward))
        self.buffer.dones.append(float(done))

    def _compute_returns_and_advantages(self):
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)

        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)

        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        return returns, advantages

    def update(self):
        if len(self.buffer) == 0:
            return None, None, None, None

        returns, advantages = self._compute_returns_and_advantages()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.tensor(np.stack(self.buffer.obs), dtype=torch.float32, device=self.device)
        global_states = torch.tensor(np.stack(self.buffer.global_states), dtype=torch.float32, device=self.device)
        action_masks = torch.tensor(np.stack(self.buffer.action_masks), dtype=torch.bool, device=self.device)
        expert_features = torch.tensor(np.stack(self.buffer.expert_features), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        sample_count = obs.size(0)
        batch_size = min(self.minibatch_size, sample_count)
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_type_balance_losses = []
        epoch_sub_balance_losses = []

        for _ in range(self.update_epochs):
            indices = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_obs = obs[batch_idx]
                batch_global = global_states[batch_idx]
                batch_masks = action_masks[batch_idx]
                batch_features = expert_features[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                logits, type_weights, sub_weights = self.actor_net(batch_obs, batch_features, batch_masks)
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic_net(batch_global).squeeze(-1)
                value_loss = F.mse_loss(values, batch_returns)

                type_uniform = torch.full(
                    (self.actor_net.num_types,),
                    1.0 / float(self.actor_net.num_types),
                    dtype=torch.float32,
                    device=self.device,
                )
                sub_uniform = torch.full(
                    (self.actor_net.sub_per_type,),
                    1.0 / float(self.actor_net.sub_per_type),
                    dtype=torch.float32,
                    device=self.device,
                )

                type_mean = type_weights.mean(dim=0)
                type_balance_loss = F.mse_loss(type_mean, type_uniform)

                sub_balance_loss = 0.0
                for t in range(self.actor_net.num_types):
                    sub_mean_t = sub_weights[:, t, :].mean(dim=0)
                    sub_balance_loss = sub_balance_loss + F.mse_loss(sub_mean_t, sub_uniform)
                sub_balance_loss = sub_balance_loss / float(self.actor_net.num_types)

                actor_loss = (
                    policy_loss
                    - self.entropy_coef * entropy
                    + self.type_balance_coef * type_balance_loss
                    + self.sub_balance_coef * sub_balance_loss
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                epoch_policy_losses.append(float(policy_loss.item()))
                epoch_value_losses.append(float(value_loss.item()))
                epoch_type_balance_losses.append(float(type_balance_loss.item()))
                epoch_sub_balance_losses.append(float(sub_balance_loss.item()))

        mean_policy_loss = float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0
        mean_value_loss = float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
        mean_type_balance = float(np.mean(epoch_type_balance_losses)) if epoch_type_balance_losses else 0.0
        mean_sub_balance = float(np.mean(epoch_sub_balance_losses)) if epoch_sub_balance_losses else 0.0

        self.policy_loss_value.append(mean_policy_loss)
        self.value_loss_value.append(mean_value_loss)
        self.type_balance_value.append(mean_type_balance)
        self.sub_balance_value.append(mean_sub_balance)
        self.buffer.clear()
        return mean_policy_loss, mean_value_loss, mean_type_balance, mean_sub_balance
