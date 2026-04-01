import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithm.MAPPO_MOE_structure.features import EXPERT_FEATURE_DIM


class MoEActorNet(nn.Module):
    """
    Mixture-of-Experts actor:
    - Expert 0: straight dash
    - Expert 1: obstacle avoidance
    - Expert 2: yielding/game-theoretic avoidance
    """

    def __init__(self, num_inputs=3, num_actions=4, expert_feature_dim=EXPERT_FEATURE_DIM, num_experts=3):
        super().__init__()
        self.num_actions = int(num_actions)
        self.num_experts = int(num_experts)
        self.expert_feature_dim = int(expert_feature_dim)

        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc_shared = nn.Linear(64 * 4 * 4, 128)
        self.expert_heads = nn.ModuleList([nn.Linear(128, self.num_actions) for _ in range(self.num_experts)])
        self.gate_head = nn.Linear(128 + self.expert_feature_dim, self.num_experts)

    def _encode_obs(self, obs_tensor):
        x = F.relu(self.conv1(obs_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc_shared(x))

    def _heuristic_expert_bias(self, feature_tensor):
        # feature index definition from features.py
        dx = feature_tensor[:, 0]
        dy = feature_tensor[:, 1]
        free = feature_tensor[:, 3:7]
        occ = feature_tensor[:, 7:11]
        near_dx = feature_tensor[:, 11]
        near_dy = feature_tensor[:, 12]
        conflict = feature_tensor[:, 14:18]

        straight = torch.stack([-dy, dx, dy, -dx], dim=1) * 2.5
        straight = straight + (free * 2.0 - 1.0) * 0.5

        obstacle = free * 2.2 - (1.0 - free) * 2.4 + straight * 0.25

        away = torch.stack([near_dy, -near_dx, -near_dy, near_dx], dim=1)
        yielding = free * 1.2 - conflict * 3.0 - occ * 2.0 + away * 1.5

        return torch.stack([straight, obstacle, yielding], dim=1)

    def _heuristic_gate_prior(self, feature_tensor):
        dist = feature_tensor[:, 2]
        occ = feature_tensor[:, 7:11]
        near_dist = feature_tensor[:, 13]
        conflict = feature_tensor[:, 14:18]
        open_ratio = feature_tensor[:, 18]

        blocked_ratio = 1.0 - open_ratio
        near_inv = 1.0 - near_dist
        adj_occ = occ.mean(dim=1)
        conflict_mean = conflict.mean(dim=1)

        straight_score = 1.4 * open_ratio + 1.2 * dist - 0.7 * near_inv - 0.8 * adj_occ
        obstacle_score = 1.8 * blocked_ratio + 0.6 * conflict_mean
        yielding_score = 1.5 * adj_occ + 1.8 * conflict_mean + 1.2 * near_inv
        return torch.stack([straight_score, obstacle_score, yielding_score], dim=1)

    def forward(self, obs_tensor, feature_tensor, action_mask=None):
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        feature_tensor = feature_tensor.float()

        shared = self._encode_obs(obs_tensor)

        expert_logits = []
        for head in self.expert_heads:
            expert_logits.append(head(shared))
        expert_logits = torch.stack(expert_logits, dim=1)  # [B, E, A]

        heuristic_bias = self._heuristic_expert_bias(feature_tensor)  # [B, E, A]
        expert_logits = expert_logits + heuristic_bias

        gate_input = torch.cat([shared, feature_tensor], dim=1)
        gate_logits = self.gate_head(gate_input) + self._heuristic_gate_prior(feature_tensor)
        gate_weights = F.softmax(gate_logits, dim=1)

        mixed_logits = torch.sum(gate_weights.unsqueeze(-1) * expert_logits, dim=1)  # [B, A]

        if action_mask is not None:
            mask = action_mask.bool()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            valid_any = mask.any(dim=1, keepdim=True)
            safe_mask = torch.where(valid_any, mask, torch.ones_like(mask, dtype=torch.bool))
            mixed_logits = mixed_logits.masked_fill(~safe_mask, -1e9)

        return mixed_logits, gate_weights, expert_logits


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


class MAPPOMoEAgent:
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
        self.moe_balance_coef = 0.02
        self.max_grad_norm = 0.5

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=1e-3)

        self.buffer = RolloutBuffer()
        self.policy_loss_value = []
        self.value_loss_value = []
        self.moe_balance_value = []

    def choose_action(self, obs, expert_features, action_mask=None, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        feature_tensor = torch.tensor(expert_features, dtype=torch.float32, device=self.device).unsqueeze(0)

        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        logits, gate_weights, _ = self.actor_net(obs_tensor, feature_tensor, mask_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), gate_weights[0].detach().cpu().numpy()

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
            return None, None, None

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
        epoch_balance_losses = []

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

                logits, gate_weights, _ = self.actor_net(batch_obs, batch_features, batch_masks)
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic_net(batch_global).squeeze(-1)
                value_loss = F.mse_loss(values, batch_returns)

                uniform = torch.full(
                    (self.actor_net.num_experts,),
                    1.0 / float(self.actor_net.num_experts),
                    dtype=torch.float32,
                    device=self.device,
                )
                gate_mean = gate_weights.mean(dim=0)
                moe_balance_loss = F.mse_loss(gate_mean, uniform)

                actor_loss = policy_loss - self.entropy_coef * entropy + self.moe_balance_coef * moe_balance_loss

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
                epoch_balance_losses.append(float(moe_balance_loss.item()))

        mean_policy_loss = float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0
        mean_value_loss = float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
        mean_balance_loss = float(np.mean(epoch_balance_losses)) if epoch_balance_losses else 0.0

        self.policy_loss_value.append(mean_policy_loss)
        self.value_loss_value.append(mean_value_loss)
        self.moe_balance_value.append(mean_balance_loss)
        self.buffer.clear()
        return mean_policy_loss, mean_value_loss, mean_balance_loss
