import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNet(nn.Module):
    def __init__(self, num_inputs=3, num_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.actions)


class MAPPOAgent:
    def __init__(self, actor_net, critic_net, device):
        self.device = device
        self.actor_net = actor_net.to(device)
        self.critic_net = critic_net.to(device)

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        # Keep convergence behavior while reducing update cost on long 30x30 episodes.
        self.update_epochs = 3
        self.minibatch_size = 256
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=1e-3)

        self.buffer = RolloutBuffer()
        self.policy_loss_value = []
        self.value_loss_value = []

    def choose_action(self, obs, action_mask=None, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor_net(obs_tensor)

        if action_mask is not None:
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            if not bool(mask_tensor.any().item()):
                mask_tensor = torch.ones_like(mask_tensor, dtype=torch.bool)
            logits = logits.masked_fill(~mask_tensor, -1e9)

        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def evaluate_value(self, global_state):
        state_tensor = torch.tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.critic_net(state_tensor)
        return float(value.item())

    def store_transition(self, obs, global_state, action_mask, action, log_prob, value, reward, done):
        self.buffer.obs.append(np.array(obs, dtype=np.float32))
        self.buffer.global_states.append(np.array(global_state, dtype=np.float32))
        self.buffer.action_masks.append(np.array(action_mask, dtype=np.float32))
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
            return None, None

        returns, advantages = self._compute_returns_and_advantages()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.tensor(np.stack(self.buffer.obs), dtype=torch.float32, device=self.device)
        global_states = torch.tensor(np.stack(self.buffer.global_states), dtype=torch.float32, device=self.device)
        action_masks = torch.tensor(np.stack(self.buffer.action_masks), dtype=torch.bool, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        sample_count = obs.size(0)
        batch_size = min(self.minibatch_size, sample_count)
        epoch_policy_losses = []
        epoch_value_losses = []

        for _ in range(self.update_epochs):
            indices = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, batch_size):
                batch_idx = indices[start:start + batch_size]

                batch_obs = obs[batch_idx]
                batch_global = global_states[batch_idx]
                batch_masks = action_masks[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]

                logits = self.actor_net(batch_obs)
                logits = logits.masked_fill(~batch_masks, -1e9)
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic_net(batch_global).squeeze(-1)
                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                epoch_policy_losses.append(float(policy_loss.item()))
                epoch_value_losses.append(float(value_loss.item()))

        mean_policy_loss = float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0
        mean_value_loss = float(np.mean(epoch_value_losses)) if epoch_value_losses else 0.0
        self.policy_loss_value.append(mean_policy_loss)
        self.value_loss_value.append(mean_value_loss)
        self.buffer.clear()
        return mean_policy_loss, mean_value_loss
