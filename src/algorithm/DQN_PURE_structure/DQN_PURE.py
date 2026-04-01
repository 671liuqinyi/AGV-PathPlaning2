from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
# import SumTree

np.set_printoptions(threshold=np.inf)

# architecture used for layout smallGrid
""" Deep Q Network """


class Net(nn.Module):
    def __init__(self, num_inputs=1, num_actions=4, map_xdim=9, map_ydim=10):
        super(Net, self).__init__()
        # structure of neural network_picture
        node_num_1 = 32
        node_num_2 = 64
        node_num_3 = 128
        node_num_4 = 256

        self.conv1 = nn.Conv2d(num_inputs, node_num_1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(node_num_1, node_num_2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(node_num_2, node_num_3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(node_num_3, node_num_4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc3 = nn.Linear(node_num_4 * int(map_xdim / 4) * int(map_ydim / 4), 256)
        self.fc4 = nn.Linear(256, num_actions)
        self.dropout = nn.Dropout(p=0.5)  # dropout during training

        # info for training

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv_pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_pool2(x)

        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        return x


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        priorities = np.array(priorities, dtype=np.float32)
        total_priority = float(self.tree.total()) + 1e-8
        sampling_probabilities = priorities / total_priority
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class Agent:
    def __init__(self, policy_net, target_net):
        # Build device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.policy_network = policy_net.to(self.device)
        self.target_network = target_net.to(self.device)

        # info for training
        self.epsilon_start = 0.2
        self.epsilon = self.epsilon_start
        self.epsilon_end = 0.95
        self.epsilon_count = 0

        # memory replay and score databases
        self.replay_mem = deque()
        self.memory_size = 50000  # replay memory capacity
        self.start_training_info_number = 500  # minimum samples before updates
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 100  # update network_picture step
        self.batch_size = 64  # memory replay batch size
        self.GAMMA = 0.98  # discount factor
        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        self.lr_start = 1e-3
        self.lr = self.lr_start
        self.lr_end = 3e-4
        self.lr_count = 0

        # self.optim = torch.optim.RMSprop(self.policy_network.parameters(), self.lr, alpha=0.95, eps=0.01)
        # self.optim = torch.optim.Adam(self.policy_network.parameters(), self.lr)
        # self.optim = torch.optim.SGD(self.policy_network.parameters(), self.lr)
        # self.optim = torch.optim.ASGD(self.policy_network.parameters(), self.lr)
        self.optim = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr, weight_decay=1e-5)
        # self.optim = torch.optim.Adadelta(self.policy_network.parameters(), self.lr)
        # self.optim = torch.optim.Adamax(self.policy_network.parameters(), self.lr)
        # self.optim = torch.optim.Rprop(self.policy_network.parameters(), self.lr)
        # self.optim = torch.optim.AdamW(self.policy_network.parameters(), self.lr)
        self.loss_function = torch.nn.SmoothL1Loss()
        self.loss_value = []

    # Inference mode action selection.
    def choose_action_test(self, obs, current_place, target_place, valid_path_matrix):
        state = torch.from_numpy(obs).float().unsqueeze(0)  # array to torch
        # state = torch.unsqueeze(state, 0)
        state = state.to(self.device)  # transform to GPU
        t_s = time.time()
        actions_value = self.policy_network.forward(state)  # get action
        t_e = time.time()
        action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
        # action = np.random.randint(0, 4)  # fully random baseline
        # action = np.array([action])
        t_ = t_e - t_s
        return action, t_

    # Training mode epsilon-greedy action selection.
    def choose_action(self, obs, current_place, target_place, valid_path_matrix, matrix_padding=0):
        if np.random.uniform() < self.epsilon:  # greedy
            state = torch.from_numpy(obs).float().unsqueeze(0)  # array to torch
            # state = torch.unsqueeze(state, 0)
            state = state.to(self.device)  # transform to GPU
            t_s = time.time()
            actions_value = self.policy_network.forward(state)  # get action
            t_e = time.time()
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()
            if not self.is_action_valid(int(action[0]), valid_path_matrix, current_place):
                action = np.array([self.find_action_safe(valid_path_matrix, current_place)])
            # action = np.random.randint(0, 4)  # fully random baseline
            # action = np.array([action])
        else:  # random explore (pure DQN, no A*)
            t_s = time.time()
            action = np.array([self.find_action_safe(valid_path_matrix, current_place)])
            t_e = time.time()
        t_ = t_e - t_s
        return action, t_

    def choose_action_as(self, current_place, target_place, valid_path_matrix, matrix_padding=0):
        action = self.find_action_safe(valid_path_matrix, current_place)
        action = np.array([action])
        return action

    def find_action_safe(self, matrix_valid_map, current_position):
        """
        Legacy-compatible fallback that only samples legal neighbors.
        It does not use A* and keeps action space within [0, 3].
        """
        action_offset = {
            0: (0, -1),  # UP
            1: (1, 0),   # RIGHT
            2: (0, 1),   # DOWN
            3: (-1, 0),  # LEFT
        }
        x, y = current_position[0], current_position[1]
        map_h = len(matrix_valid_map)
        map_w = len(matrix_valid_map[0])
        valid_actions = []

        for action, (dx, dy) in action_offset.items():
            nx, ny = x + dx, y + dy
            if nx < 1 or ny < 1 or nx > map_w or ny > map_h:
                continue
            if matrix_valid_map[ny - 1][nx - 1] != 0:
                valid_actions.append(action)

        if not valid_actions:
            return np.random.randint(0, 4)
        return random.choice(valid_actions)

    def is_action_valid(self, action, matrix_valid_map, current_position):
        action_offset = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
        }
        if action not in action_offset:
            return False
        dx, dy = action_offset[action]
        nx = current_position[0] + dx
        ny = current_position[1] + dy
        map_h = len(matrix_valid_map)
        map_w = len(matrix_valid_map[0])
        if nx < 1 or ny < 1 or nx > map_w or ny > map_h:
            return False
        return matrix_valid_map[ny - 1][nx - 1] != 0

    def store_transition(self, s, a, r, s_, is_done):
        """store experience in a 'prioritized replay' way"""
        """value by prediction"""
        with torch.no_grad():
            state = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            target = self.policy_network.forward(state).cpu()
            a = int(a)
            old_val = target[0][a]
            next_state = torch.from_numpy(s_).float().unsqueeze(0).to(self.device)
            target_val = self.target_network.forward(next_state)
            if is_done == 1:
                new_val = r
            else:
                new_val = r + self.GAMMA * torch.max(target_val)
        """difference between old_val and new_val"""
        error = abs(old_val - new_val).cpu()
        error = error.detach().numpy()
        self.memory.add(error, (np.array(s), a, r, np.array(s_), is_done))

        if self.memory.tree.n_entries >= self.start_training_info_number:
            self.update_network()

    def update_network(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:  # sync every TARGET_REPLACE_ITER steps
            self.target_network.load_state_dict(self.policy_network.state_dict())
        self.learn_step_counter += 1

        # Sample a prioritized mini-batch.
        # batch = random.sample(self.replay_mem, self.batch_size)
        batch, idxs, is_weights = self.memory.sample(self.batch_size)
        batch_s, batch_a, batch_r, batch_n, batch_is_done = zip(*batch)

        # convert from numpy to pytorch
        batch_s = torch.from_numpy(np.stack(batch_s)).float().to(self.device)  # .to(torch.float32)
        # print("batch_s", batch_s)
        batch_r = torch.Tensor(batch_r).unsqueeze(1).to(self.device)
        # print(batch_r)
        # print("batch_a", batch_a)
        batch_a = torch.LongTensor(batch_a).unsqueeze(1).to(self.device)
        # print("batch_a_", batch_a)
        batch_n = torch.from_numpy(np.stack(batch_n)).float().to(self.device)
        batch_is_done = torch.LongTensor(batch_is_done).unsqueeze(1).to(self.device)
        is_weights = torch.from_numpy(np.array(is_weights, dtype=np.float32)).unsqueeze(1).to(self.device)

        state_action_values = self.policy_network(batch_s).gather(1, batch_a)

        # Estimate next-state values.
        next_state_values = self.target_network(batch_n)
        # Compute the expected Q values
        next_state_values = next_state_values.detach().max(1)[0]
        next_state_values = next_state_values.unsqueeze(1)
        # Double DQN target branch.
        next_state_values_ = self.target_network(batch_n)
        next_state_values__ = self.policy_network(batch_n)

        # print("torch.max(next_state_values__, 1)[1].unsqueeze(1)", torch.max(next_state_values__, 1)[1].unsqueeze(1))
        next_state_values___ = next_state_values_.gather(1, torch.max(next_state_values__, 1)[1].unsqueeze(1))
        # print("next_state_values___", next_state_values___)

        # expected_state_action_values = (next_state_values * self.GAMMA)*(1-batch_is_done) + batch_r
        expected_state_action_values = (next_state_values___ * self.GAMMA) * (1 - batch_is_done) + batch_r  # DDQN

        # calculate loss
        # print("state_action_values", state_action_values)
        # print("expected_state_action_values", expected_state_action_values)
        loss_per_sample = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        loss = (loss_per_sample * is_weights).mean()
        # optimize model - update weights
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=5.0)
        self.optim.step()

        # store loss value
        if loss.item() >= 0.5:
            self.loss_value.append(0.5)
        else:
            self.loss_value.append(loss.item())

    def change_learning_rate(self, times):
        if self.lr_count == times:
            print("the value of current learning rate is {}".format(self.lr))
        if self.lr_count > times:
            return
        else:
            self.lr = self.lr - (self.lr_start - self.lr_end) / times
            self.lr = max(self.lr, self.lr_end)
            for param_group in self.optim.param_groups:
                param_group["lr"] = self.lr
        self.lr_count += 1

    def change_explore_rate(self, times):
        if self.epsilon_count >= times:
            self.epsilon = self.epsilon_end
        else:
            self.epsilon = self.epsilon + (self.epsilon_end - self.epsilon_start) / times
        self.epsilon_count += 1

        if self.epsilon_count == times:
            print("exploring rate is 1.")
