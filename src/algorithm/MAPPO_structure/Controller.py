import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import algorithm.Manager.SaveManager as saveManager
from algorithm.MAPPO_structure.MAPPO import ActorNet, CriticNet, MAPPOAgent
from algorithm.Manager.StateManager import StateManager
from utils.path_utils import get_next_run_dir

np.set_printoptions(threshold=np.inf)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


class MAPPOAgentController:
    """Bridge between Scene and MAPPO algorithm."""

    def __init__(
        self,
        rmfs_scene,
        map_xdim,
        map_ydim,
        max_task,
        control_mode="train_NN",
        state_number=3,
        curriculum_stages=None,
    ):
        # Fix save/load path to src/... regardless of process working directory.
        self.base_save_path = os.path.join(SRC_DIR, "algorithm", "MAPPO_structure", "423421")
        os.makedirs(self.base_save_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_save_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.base_save_path, "test"), exist_ok=True)

        if control_mode == "train_NN":
            self.current_save_dir = get_next_run_dir(os.path.join(self.base_save_path, "train"))
        else:
            self.current_save_dir = os.path.join(self.base_save_path, "test")

        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-log.txt")

        print("start simulation with MAPPO algorithm")
        print("map_xdim:", map_xdim, "map_ydim:", map_ydim, "state_number:", state_number)
        print("save_dir:", self.current_save_dir)

        self.control_mode = control_mode
        self.state_number = state_number
        self.rmfs_model = rmfs_scene
        self.map_xdim = map_xdim
        self.map_ydim = map_ydim
        self.max_agents = len(self.rmfs_model.explorer_group)

        self.state_manager = StateManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_state_dim = self.map_xdim * self.map_ydim + self.max_agents * 5
        self.agent = None
        self.create_agent()

        self.simulation_times = 800 if self.control_mode == "train_NN" else 1
        self.max_value = max_task * 3
        self.max_value_times = 0
        self.duration_times = 30
        self.curriculum_stages = curriculum_stages if curriculum_stages else []
        self.current_stage_index = -1

        self.lr_start_decay = False
        self.lifelong_reward = []
        self.action_length_record = 0

        self.reward_acc = 0
        self.veh_group = []
        self.logs = []

    def create_agent(self):
        if self.control_mode == "train_NN":
            print("create NN")
            actor_net = ActorNet(self.state_number, self.rmfs_model.action_number)
            critic_net = CriticNet(self.global_state_dim)
        else:
            print("load NN")
            actor_path = self._resolve_model_path("RMFS_MAPPO_actor_net.pt")
            critic_path = self._resolve_model_path("RMFS_MAPPO_critic_net.pt")
            actor_net = torch.load(actor_path, map_location=self.device)
            critic_net = torch.load(critic_path, map_location=self.device)

        self.agent = MAPPOAgent(actor_net, critic_net, self.device)

    def self_init(self):
        self.reward_acc = 0
        self.veh_group = []
        self.action_length_record = 0
        self.agent.buffer.clear()

    def model_run(self, run_mode):
        print("model is controlled by neural network")

        for i_episode in range(self.simulation_times):
            self._apply_curriculum(i_episode)
            start_time = time.time()
            self.self_init()
            self.rmfs_model.init()

            render = run_mode == "use_NN"
            _ = self.rmfs_model.run_game(control_pattern="intelligent", smart_controller=self, render=render)

            if run_mode == "train_NN" and len(self.agent.buffer) > 0:
                self.agent.update()

            episode_cost = time.time() - start_time
            self.lifelong_reward.append(self.reward_acc)
            log = (
                "i_episode {}\treward_accu: {} \taction_length: {} \tepisode_cost: {:.2f}s".format(
                    i_episode, self.reward_acc, self.action_length_record, episode_cost
                )
            )
            self.logs.append(log)
            self.save_log(log)
            print(log)

            if run_mode == "train_NN" and (i_episode + 1) % 100 == 0:
                self.save_neural_network(auto=True)

            if run_mode == "train_NN" and self.check_determination(self.reward_acc):
                break

        if run_mode == "train_NN":
            self.save_neural_network(auto=False)
            saveManager.draw_picture(
                self.lifelong_reward,
                title="Cumulative Reward",
                x_label="training episodes",
                y_label="cumulative reward",
                color="g",
                save_path=os.path.join(self.current_save_dir, "Cumulative Reward"),
                smooth=True,
            )
            saveManager.draw_picture(
                self.agent.policy_loss_value,
                title="Policy Loss",
                x_label="training episodes",
                y_label="loss",
                color="b",
                save_path=os.path.join(self.current_save_dir, "Policy Loss"),
            )
            saveManager.draw_picture(
                self.agent.value_loss_value,
                title="Value Loss",
                x_label="training episodes",
                y_label="loss",
                color="k",
                save_path=os.path.join(self.current_save_dir, "Value Loss"),
            )
            plt.show()

    def _apply_curriculum(self, i_episode):
        if self.control_mode != "train_NN" or not self.curriculum_stages:
            return

        stage_index = len(self.curriculum_stages) - 1
        for idx, stage in enumerate(self.curriculum_stages):
            end_episode = int(stage.get("end_episode", 0))
            if i_episode < end_episode:
                stage_index = idx
                break

        stage = self.curriculum_stages[stage_index]
        if stage_index != self.current_stage_index:
            self.current_stage_index = stage_index
            print(
                "curriculum stage {} => task_num_limit={}, max_steps={}".format(
                    stage_index + 1, stage.get("task_num_limit"), stage.get("max_steps")
                )
            )

        task_num_limit = stage.get("task_num_limit")
        max_steps = stage.get("max_steps")

        if hasattr(self.rmfs_model, "layout") and hasattr(self.rmfs_model.layout, "task_num_limit"):
            self.rmfs_model.layout.task_num_limit = task_num_limit
        if hasattr(self.rmfs_model, "max_training_steps") and max_steps is not None:
            self.rmfs_model.max_training_steps = int(max_steps)

    def choose_action(self, all_info, this_veh):
        veh_obj = self._get_veh_obj(this_veh)

        obs, this_veh_cp, _, valid_path_matrix = self.state_manager.create_state(all_info, this_veh, obs_clip=True)
        obs = np.array(obs, dtype=np.float32)
        global_state = self.create_global_state(all_info)
        action_mask = self.create_action_mask(valid_path_matrix, this_veh_cp)

        deterministic = self.control_mode == "use_NN"
        action, log_prob = self.agent.choose_action(obs, action_mask=action_mask, deterministic=deterministic)
        value = self.agent.evaluate_value(global_state)

        veh_obj.obs_current = obs
        veh_obj.global_state = global_state
        veh_obj.action_mask = action_mask
        veh_obj.action = action
        veh_obj.log_prob = log_prob
        veh_obj.value = value

        self.action_length_record += 1
        return action

    def store_info(self, all_info, reward, is_end, this_veh):
        self.reward_acc += reward
        if self.control_mode == "use_NN":
            return

        veh_obj = self._get_veh_obj(this_veh)
        if veh_obj.action is None:
            return

        self.agent.store_transition(
            obs=veh_obj.obs_current,
            global_state=veh_obj.global_state,
            action_mask=veh_obj.action_mask,
            action=veh_obj.action,
            log_prob=veh_obj.log_prob,
            value=veh_obj.value,
            reward=reward,
            done=1 if is_end else 0,
        )

        veh_obj.clear_cache()

        if is_end:
            self.agent.update()

    def create_global_state(self, all_info):
        layout_flat = np.array(all_info[0], dtype=np.float32).reshape(-1) / 3.0
        veh_matrix = np.zeros((self.max_agents, 5), dtype=np.float32)

        valid_num = min(len(all_info) - 1, self.max_agents)
        for idx in range(valid_num):
            _, current_place, target_place, loaded = all_info[idx + 1]
            veh_matrix[idx, 0] = float(current_place[0]) / max(self.map_xdim, 1)
            veh_matrix[idx, 1] = float(current_place[1]) / max(self.map_ydim, 1)
            veh_matrix[idx, 2] = float(target_place[0]) / max(self.map_xdim, 1)
            veh_matrix[idx, 3] = float(target_place[1]) / max(self.map_ydim, 1)
            veh_matrix[idx, 4] = float(loaded)

        return np.concatenate([layout_flat, veh_matrix.reshape(-1)]).astype(np.float32)

    def create_action_mask(self, valid_path_matrix, current_place):
        action_num = self.rmfs_model.action_number
        action_mask = np.zeros(action_num, dtype=np.float32)
        action_to_offset = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

        map_ydim = len(valid_path_matrix)
        map_xdim = len(valid_path_matrix[0])

        for action in range(action_num):
            dx, dy = action_to_offset[action]
            next_x = current_place[0] + dx
            next_y = current_place[1] + dy
            if 1 <= next_x <= map_xdim and 1 <= next_y <= map_ydim:
                if valid_path_matrix[next_y - 1][next_x - 1] != 0:
                    action_mask[action] = 1.0

        if action_mask.sum() == 0:
            action_mask[:] = 1.0

        return action_mask

    def check_determination(self, reward_accu):
        if reward_accu >= self.max_value - 1:
            self.max_value_times += 1
            self.lr_start_decay = True
        else:
            self.max_value_times = 0

        return self.max_value_times == self.duration_times

    def save_neural_network(self, auto=False):
        if self.control_mode == "use_NN":
            return

        if auto:
            torch.save(self.agent.actor_net, os.path.join(self.current_save_dir, "RMFS_MAPPO_actor_net_auto.pt"))
            torch.save(self.agent.critic_net, os.path.join(self.current_save_dir, "RMFS_MAPPO_critic_net_auto.pt"))
        else:
            torch.save(self.agent.actor_net, os.path.join(self.current_save_dir, "RMFS_MAPPO_actor_net.pt"))
            torch.save(self.agent.critic_net, os.path.join(self.current_save_dir, "RMFS_MAPPO_critic_net.pt"))

    def save_log(self, content=""):
        log_path = os.path.join(self.current_save_dir, self.time_str)
        with open(log_path, "a") as f:
            f.write(content)
            f.write("\r\n")

    def _get_veh_obj(self, this_veh):
        for veh in self.veh_group:
            if veh.veh_name == this_veh:
                return veh

        new_veh = VehObj(this_veh)
        self.veh_group.append(new_veh)
        return new_veh

    def _resolve_model_path(self, model_name):
        preferred_path = os.path.join(self.current_save_dir, model_name)
        if os.path.exists(preferred_path):
            return preferred_path

        train_dir = os.path.join(self.base_save_path, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError("No trained MAPPO model found in test or train folders.")

        latest_run = None
        latest_num = -1
        for dir_name in os.listdir(train_dir):
            if not dir_name.startswith("run"):
                continue
            num_str = dir_name.replace("run", "", 1)
            if not num_str.isdigit():
                continue
            checkpoint_path = os.path.join(train_dir, dir_name, model_name)
            if os.path.exists(checkpoint_path):
                run_num = int(num_str)
                if run_num > latest_num:
                    latest_num = run_num
                    latest_run = checkpoint_path

        if latest_run is None:
            raise FileNotFoundError("No trained MAPPO model found in test or train folders.")

        return latest_run


class VehObj:
    def __init__(self, this_veh):
        self.veh_name = this_veh
        self.obs_current = None
        self.global_state = None
        self.action_mask = None
        self.action = None
        self.log_prob = None
        self.value = None

    def clear_cache(self):
        self.obs_current = None
        self.global_state = None
        self.action_mask = None
        self.action = None
        self.log_prob = None
        self.value = None
