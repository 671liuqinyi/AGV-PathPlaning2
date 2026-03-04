import datetime
import os
import time

from algorithm.DQN_structure.DQN import Net as Net
from algorithm.DQN_structure.DQN import Agent as Agent
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.path_utils import get_next_run_dir

np.set_printoptions(threshold=np.inf)


# save_path = 'algorithm/DQN_structure/323321'
# save_path = 'algorithm/DQN_structure/423421'
# save_path = 'algorithm/DQN_structure/523521'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
#     os.makedirs(save_path + '/train')
#     os.makedirs(save_path + '/test')


class DQNAgentController:
    """
        a link between environment and algorithm
        """

    # state_number要修改
    # marix_padding要删除
    def __init__(self, rmfs_scene, map_xdim, map_ydim, max_task, control_mode=1, state_number=4):

        # ========== 新增：动态设置基础路径 + 生成run文件夹 ==========
        self.base_save_path = 'algorithm/DQN_structure/323321'  # 可替换为[323321/423421/523521]
        # 确保基础目录存在
        os.makedirs(self.base_save_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_save_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.base_save_path, 'test'), exist_ok=True)

        # 训练模式下生成递增的run文件夹，测试模式沿用原有test目录
        if control_mode == "train_NN":
            self.train_run_dir = get_next_run_dir(os.path.join(self.base_save_path, 'train'))
            self.current_save_dir = self.train_run_dir  # 训练时所有文件保存到runN
        else:
            self.current_save_dir = os.path.join(self.base_save_path, 'test')  # 测试仍用test目录

        # 获取当前时间并格式化为指定样式
        current_time = datetime.datetime.now()
        # 格式化：年-月-日--时-分-log.txt
        self.time_str = current_time.strftime("%Y-%m-%d_%H-%M-log.txt")

        print("start simulation with DQN algorithm")
        print("map_xdim:", map_xdim, "map_ydim:", map_ydim, "state_number:", state_number)

        '''received parameters'''
        self.control_mode = control_mode
        self.state_number = state_number

        '''get RMFS object'''
        self.rmfs_model = rmfs_scene

        '''create/load neural network_picture'''
        policy_net, target_net = None, None
        if self.control_mode == "train_NN":
            print("create NN")
            policy_net = Net(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
            target_net = Net(self.state_number, self.rmfs_model.action_number, map_xdim, map_ydim)
        elif self.control_mode == "use_NN":
            print("load NN")
            policy_net = torch.load(f'{self.current_save_dir}/RMFS_DQN_policy_net.pt')
            target_net = torch.load(f'{self.current_save_dir}/RMFS_DQN_target_net.pt')

        '''create Agent object'''
        self.agent = Agent(policy_net, target_net)
        ############################################# control_mode是否还在使用

        '''training parameters'''
        self.simulation_times = 800
        self.max_value = max_task * 3
        self.max_value_times = 0
        self.duration_times = 30
        self.interupt_num = 0
        self.interupt_times = 0
        #############################################
        self.acc_max = 0
        self.acc_max_val = 0  # 40

        self.lr_start_decay = False

        self.lifelong_reward = []
        self.action_length_record = 0
        self.time_list = []

        """":parameter"""
        self.reward_acc = 0
        self.veh_group = []
        self.logs = []

    def self_init(self):
        self.reward_acc = 0
        self.veh_group = []
        self.action_length_record = 0
        self.time_list = []

    def model_run(self, run_mode):  # mainloop for training/running
        print("model is controlled by neural network")

        for i_episode in range(self.simulation_times):
            # 统计每一轮训练时间
            start_time = time.time()
            self.self_init()
            self.rmfs_model.init()
            # print("i_episode", i_episode)
            """"transfer the controller to the model"""
            """the model runs once"""
            running_time = self.rmfs_model.run_game(control_pattern="intelligent", smart_controller=self, render=False)
            # running_time = self.rmfs_model.run_game(control_pattern="intelligent", smart_controller=self, render=True)
            end_time = time.time()
            episode_cost = end_time - start_time  # 本轮耗时（秒）
            self.lifelong_reward.append(self.reward_acc)
            log = 'i_episode {}\treward_accu: {} \taction_length: {} \tepisode_cost: {:.2f}s'.format(i_episode,
                                                                                                     self.reward_acc,
                                                                                                     self.action_length_record,
                                                                                                     episode_cost)
            self.logs.append(log)
            self.save_log(log)
            print(log)
            # print("self.time_list", np.array(self.time_list).sum())
            # print("running_time", running_time)
            # 改变探索率
            if self.lr_start_decay:
                self.agent.change_learning_rate(times=200)
                # 改变lr
                self.agent.change_explore_rate(times=200)

            if run_mode == 'train_NN' and (i_episode + 1) % 100 == 0:
                self.save_neural_network(auto=True)
                # check whether determination condition meets
            if self.check_determination(self.reward_acc):
                break
        if run_mode == 'train_NN':
            self.save_neural_network(auto=False)
            self.draw_picture(self.lifelong_reward, p_title="Cumulative Reward", p_xlabel="training episodes",
                              p_ylabel="cumulative reward", p_color="g", )
            self.draw_picture(self.agent.loss_value, p_title="Loss Value", p_xlabel="Training steps",
                              p_ylabel="Loss value",
                              p_color="k", )
        # self.save_log(run_mode=run_mode)
        plt.show()

    def choose_action(self, all_info, this_veh):  # all_infor=[layout  , current_place, target_place
        """build a VehObj to store information"""
        veh_found = False
        veh_obj = None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_found = True
                veh_obj = veh
                break
        if not veh_found:
            veh_obj = VehObj(this_veh)
            self.veh_group.append(veh_obj)

        """get observation and other info"""
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.create_state(all_info, this_veh)
        """get action"""
        veh_obj.obs_current = obs
        veh_obj.obs_valid_matrix = valid_path_matrix
        action_l, t_ = self.agent.choose_action(obs, current_place=this_veh_cp, target_place=this_veh_tp,
                                                valid_path_matrix=valid_path_matrix)  # state should be formatted as array
        action = action_l[0]
        """record info"""
        self.time_list.append(t_)
        veh_obj.action.append(action)
        self.action_length_record += 1
        return action

    def check_action(self, this_veh_cp, veh_tp, valid_path_matrix, action, this_veh):
        # print("this_veh", this_veh)
        # print("valid_path_matrix", valid_path_matrix)
        # print("this_veh_cp", this_veh_cp)
        # print("this_veh_tp", veh_tp)
        # print("action", action)
        veh_cp = this_veh_cp.copy()
        if action == 0:
            veh_cp[1] -= 1
        if action == 1:
            veh_cp[0] += 1
        if action == 2:
            veh_cp[1] += 1
        if action == 3:
            veh_cp[0] -= 1

        if veh_cp[0] <= 0 or veh_cp[1] <= 0 or veh_cp[0] > len(valid_path_matrix[0]) or veh_cp[1] > len(
                valid_path_matrix):
            action = self.agent.choose_action_as(current_place=this_veh_cp, target_place=veh_tp,
                                                 valid_path_matrix=valid_path_matrix)[
                0]  # state should be formatted as array
        else:
            if valid_path_matrix[veh_cp[1] - 1][veh_cp[0] - 1] == 0:
                action = self.agent.choose_action_as(current_place=this_veh_cp, target_place=veh_tp,
                                                     valid_path_matrix=valid_path_matrix)[
                    0]  # state should be formatted as array
        # print("action", action)
        return action

    def store_info(self, all_info, reward, is_end, this_veh):
        self.reward_acc += reward
        if self.control_mode == "use_NN":
            """using NN, no need to store info and train NN"""
            return

        veh_obj = None
        for veh in self.veh_group:
            if this_veh == veh.veh_name:
                veh_obj = veh
                break
        obs, this_veh_cp, this_veh_tp, valid_path_matrix = self.create_state(all_info, this_veh)
        # 先存储经验，在保留新数据
        veh_obj.obs_next, veh_obj.reward = obs, reward

        is_done = 1 if is_end else 0
        self.agent.store_transition(veh_obj.obs_current, veh_obj.action[-1], veh_obj.reward, veh_obj.obs_next, is_done)

    def create_state(self, all_info, this_veh):
        layout = all_info[0]
        occupied_place = []
        occupied_target = []
        current_place = 0
        target_place = 0
        veh_loaded = False
        """obtain information about current_place, target_place, occupied_place, occupied_target"""
        for i in range(1, len(all_info)):
            one_veh = all_info[i]
            veh_name_, current_place_, target_place_, veh_loaded_ = one_veh[0], one_veh[1], one_veh[2], one_veh[3]
            if veh_name_ == this_veh:  # target_veh
                current_place, target_place, veh_loaded = current_place_, target_place_, veh_loaded_
            else:
                occupied_place.append(current_place_)
                occupied_target.append(target_place_)
        """"format observations"""
        valid_path_matrix, forbidden_path_matrix, basic_matrix_array = \
            self.create_path_matrix(layout, veh_loaded, current_place, target_place, occupied_place)
        current_position_matrix, target_position_matrix, occupied_position_matrix \
            = self.create_position_matrix(layout, current_place, target_place, occupied_place, occupied_target)

        state = np.array((current_position_matrix, target_position_matrix, valid_path_matrix))
        """neural network uses state to make decision"""
        """astar algorithm uses current_place, target_place, valid_path_matrix to make decision"""
        return state, current_place, target_place, valid_path_matrix

    def create_path_matrix(self, layout, veh_loaded, current_place, target_place, occupied_place):
        # valid_path_matrix, forbidden_path_matrix
        valid_path, valid_path_one_line = [], []
        forbidden_path, forbidden_path_one_line = [], []

        # 制作原始的valid_path和forbidden_path
        for map_one_line in layout:
            for one_cell in map_one_line:
                if one_cell == 0:
                    valid_path_one_line.append(1.)
                    forbidden_path_one_line.append(0.)
                elif one_cell == 1:
                    if veh_loaded == 0:
                        valid_path_one_line.append(1.)
                        forbidden_path_one_line.append(0.)
                    else:
                        valid_path_one_line.append(0.)
                        forbidden_path_one_line.append(1.)
                elif one_cell == 2:
                    valid_path_one_line.append(0.)
                    forbidden_path_one_line.append(1.)
                else:
                    print("create_path_matrix:wrong matrix")
            valid_path.append(valid_path_one_line)
            valid_path_one_line = []
            forbidden_path.append(forbidden_path_one_line)
            forbidden_path_one_line = []

        valid_path_matrix = np.array(valid_path)
        forbidden_path_matrix = np.array(forbidden_path)

        valid_path_matrix_o = valid_path_matrix.copy()
        forbidden_path_matrix_o = forbidden_path_matrix.copy()

        """调整valid_path_matrix和forbidden_path_matrix"""
        # 根据current_position和target_position调整
        valid_path_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0  # current_place_array样式[[x],[y]]
        forbidden_path_matrix[current_place[1] - 1][current_place[0] - 1] = 0.0
        valid_path_matrix[target_place[1] - 1][target_place[0] - 1] = 1.0  # current_place_array样式[[x],[y]]
        forbidden_path_matrix[target_place[1] - 1][target_place[0] - 1] = 0.0

        # useless?
        valid_path_matrix_o[current_place[1] - 1][current_place[0] - 1] = 2.0  # current_place_array样式[[x],[y]]
        valid_path_matrix_o[target_place[1] - 1][target_place[0] - 1] = 3.0  # current_place_array样式[[x],[y]]

        # 其他车辆对道路的占用
        if occupied_place:
            for o_place in occupied_place:
                valid_path_matrix[o_place[1] - 1][o_place[0] - 1] = 0.0  # current_place_array样式[[x],[y]]
                forbidden_path_matrix[o_place[1] - 1][o_place[0] - 1] = 1.0
                valid_path_matrix_o[o_place[1] - 1][o_place[0] - 1] = 4.0  # current_place_array样式[[x],[y]]

        """"无效"""
        # 赋予更高权重
        basic_matrix = self.create_basic_matrix(layout)
        basic_matrix_array = np.array(basic_matrix)

        #
        current_p_x, current_p_y = current_place[0] - 1, current_place[1] - 1
        up, right, down, left = (0, -1), (1, 0), (0, 1), (-1, 0)
        four_dict = [up, right, down, left]
        direction_length = 3

        if valid_path_matrix[current_p_y][current_p_x] != 0:
            basic_matrix_array[current_p_y][current_p_x] = 1.0  # current_place_array样式[[x],[y]]
            for one_direction in four_dict:
                pos = [current_p_x + one_direction[0], current_p_y + one_direction[1]]
                pos_further = [current_p_x + one_direction[0] * 2, current_p_y + one_direction[1] * 2]
                if pos[0] < 0 or pos[1] < 0 or pos[0] >= len(valid_path_matrix[0]) or pos[1] >= len(valid_path_matrix):
                    continue
                else:
                    if valid_path_matrix[pos[1]][pos[0]] != 0:
                        basic_matrix_array[pos[1]][pos[0]] = 1.0  # current_place_array样式[[x],[y]]
                        # 当前位置有效，探索更远一格位置
                    elif valid_path_matrix[pos[1]][pos[0]] == 0:
                        basic_matrix_array[pos[1]][pos[0]] = -1.0  # current_place_array样式[[x],[y]]

        return valid_path_matrix, forbidden_path_matrix, basic_matrix_array

    def create_basic_matrix(self, layout):
        basic_matrix, basic_matrix_one_line = [], []
        for map_one_line in layout:
            for one_cell in map_one_line:
                basic_matrix_one_line.append(0.)
            basic_matrix.append(basic_matrix_one_line)
            basic_matrix_one_line = []
        return basic_matrix

    def create_position_matrix(self, layout, current_place, target_place, occupied_place, occupied_target):
        basic_matrix = self.create_basic_matrix(layout)
        basic_matrix_array = np.array(basic_matrix)

        # 构建current_position_matrix
        current_position_matrix = basic_matrix_array.copy()
        current_position_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0

        # 构建target_position_matrix
        target_position_matrix = basic_matrix_array.copy()
        target_position_matrix[target_place[1] - 1][target_place[0] - 1] = 1.0

        occupied_position_matrix = basic_matrix_array.copy()
        if occupied_place:
            for occupied_ in occupied_place:
                occupied_position_matrix[occupied_[1] - 1][occupied_[0] - 1] = 1.0

        return current_position_matrix, target_position_matrix, occupied_position_matrix

    def draw_picture(self, p_data, p_title="NoTitle", p_xlabel="xlabel", p_ylabel="ylabel", p_color="g"):
        plt.figure(figsize=(16, 9))  # 调整长宽比
        plt.title(p_title)
        plt.xlabel(p_xlabel)
        plt.ylabel(p_ylabel)
        plt.plot(p_data, color=p_color)
        plt.tight_layout()  # 去除白边
        plt.savefig(f'{self.current_save_dir}/' + p_title, dpi=300)  # 设置存储格式和分辨率

    def check_determination(self, reward_accu):
        # check whether the determination meets
        if reward_accu >= self.max_value - 1:
            self.acc_max += 1
            self.max_value_times = self.max_value_times + 1
            self.lr_start_decay = True  # lr start to decay
            if self.interupt_times > self.interupt_num:
                self.interupt_times = 0
        else:
            if self.interupt_times >= self.interupt_num:
                self.max_value_times = 0
            else:
                pass
            self.interupt_times += 1

            # self.interupt_num = 1
            # self.interupt_times = 0

        if self.max_value_times == self.duration_times:
            return True
        else:
            return False

    def save_neural_network(self, auto=False, run_mode=''):
        # 非训练模式不保存
        # if run_mode != 'train_NN':
        #     return
        if auto:
            print("neural network auto-save")
            print(os.getcwd())
            torch.save(self.agent.policy_network, f'{self.current_save_dir}/RMFS_DQN_policy_net_auto.pt')
            torch.save(self.agent.target_network, f'{self.current_save_dir}/RMFS_DQN_target_net_auto.pt')
        else:
            torch.save(self.agent.policy_network, f'{self.current_save_dir}/RMFS_DQN_policy_net.pt')
            torch.save(self.agent.target_network, f'{self.current_save_dir}/RMFS_DQN_target_net.pt')

    def save_log(self, content=""):
        log_path = os.path.join(self.current_save_dir, self.time_str)
        with open(log_path, 'a') as f:
            f.write(content)
            f.write("\r\n")


class VehObj:
    """veh object"""

    def __init__(self, this_veh):
        self.veh_name = this_veh
        """"逐个检查"""
        # self.obs_list = []
        self.obs_current = 0
        self.obs_next = 0
        self.obs_forbidden_matrix = 0
        self.obs_valid_matrix = 0
        self.action = []
        self.reward = 0
        self.is_end = False
        self.last_state = 0
        self.last_state_store = False
