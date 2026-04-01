import time

from multiAGVscene.Layout import Layout  # layout
from multiAGVscene.Explorer import Explorer  # explorer
from multiAGVscene.Scene import Scene  # Scene
from algorithm.Manager.ExpertManager import Expert as Expert
from algorithm.DQN_structure.Controller import DQNAgentController as modelController
import torch
import numpy as np
from algorithm.DQN_structure.DQN import Agent as Agent


def create_test_task_info(layout, explorer_group):
    """
    构造测试任务的全量信息（与Controller.create_state入参格式一致）
    :param layout: 场景布局对象
    :param explorer_group: AGV列表
    :return: all_info - 格式：[layout矩阵, AGV1信息, AGV2信息,...]
    """
    # 1. 获取场景布局矩阵（0=可通行，1=货架，2=障碍）
    layout_matrix = layout.layout

    # 2. 构造AGV任务信息（单AGV示例，多AGV可扩展）
    all_info = [layout_matrix]  # 第一个元素是布局矩阵
    for explorer in explorer_group:
        veh_name = explorer.explorer_name
        # 自定义测试任务：初始位置+目标位置+载货状态
        current_place = [1, 1]  # AGV初始位置（避开障碍/货架）
        target_place = [10, 12]  # AGV目标位置
        veh_loaded = 1  # 0=未载货，1=载货
        all_info.append([veh_name, current_place, target_place, veh_loaded])

    return all_info


def run_dqn_model_test1():
    """
    完整的DQN模型测试流程：场景初始化→加载模型→生成测试任务→执行动作预测→验证结果
    """
    # ===================== 1. 初始化测试场景 =====================
    ss_x_width, ss_y_width, ss_x_num, ss_y_num, ps_num = 3, 2, 3, 3, 2
    layout_list = None
    task_list = None
    # 创建布局（与训练时参数一致）
    layout = Layout(storage_station_x_width=ss_x_width, storage_station_y_width=ss_y_width,
                    storage_station_x_num=ss_x_num, storage_station_y_num=ss_y_num,
                    picking_station_number=ps_num, layout_list=layout_list, task_list=task_list)
    # 创建AGV（单AGV测试）
    explorer_num = 1
    explorer_group = []
    for i in range(explorer_num):
        veh_name = "veh" + str(i + 1)
        explorer = Explorer(layout, veh_name=veh_name, icon_name=veh_name)
        explorer_group.append(explorer)
    # 创建场景
    multi_agv_scene = Scene(layout, explorer_group)

    # ===================== 2. 加载预训练DQN模型 =====================
    # 模型路径（替换为实际训练保存的路径）
    policy_net_path = 'algorithm/DQN_structure/323321/train/run1/RMFS_DQN_policy_net.pt'
    target_net_path = 'algorithm/DQN_structure/323321/train/run1/RMFS_DQN_target_net.pt'

    # 加载模型（指定CPU/GPU，避免设备不匹配）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = torch.load(policy_net_path, map_location=device)
    target_net = torch.load(target_net_path, map_location=device)

    # 切换为评估模式（禁用Dropout/BatchNorm，避免推理结果波动）
    policy_net.eval()
    target_net.eval()

    # 初始化Agent（与训练时参数一致）
    agent = Agent(policy_net, target_net)
    print("✅ 预训练模型加载完成，设备：", device)

    # ===================== 3. 生成测试任务 =====================
    all_info = create_test_task_info(layout, explorer_group)
    test_veh_name = "veh1"  # 测试的AGV名称
    print(f"\n📌 测试任务信息：")
    print(f"AGV名称：{test_veh_name}")
    print(f"初始位置：{all_info[1][1]} | 目标位置：{all_info[1][2]} | 载货状态：{all_info[1][3]}")
    print(f"场景布局矩阵：\n{np.array(all_info[0])}")

    # ===================== 4. 初始化DQN控制器（复用核心逻辑） =====================
    # 控制器用于复用create_state、check_action等核心方法
    # controller = modelController(
    #     rmfs_scene=multi_agv_scene,
    #     map_xdim=layout.scene_x_width,
    #     map_ydim=layout.scene_y_width,
    #     max_task=len(layout.storage_station_list),
    #     control_mode="use_NN",  # 测试模式
    #     state_number=3  # 与Controller中create_state输出的state维度一致（3个矩阵）
    # )
    # 替换控制器的Agent为加载的预训练Agent
    # controller.agent = agent

    # ===================== 5. 执行模型测试（动作预测） =====================
    # 5.1 生成模型输入的观测（obs）
    obs, this_veh_cp, this_veh_tp, valid_path_matrix = create_state(all_info, this_veh=test_veh_name)
    # current_position_matrix, target_position_matrix, valid_path_matrix = obs
    print(f"\n🔍 模型输入观测维度：{obs.shape}")

    # 5.2 调用Agent选择动作（模拟推理过程）
    action_l, infer_time = agent.choose_action_test(obs, this_veh_cp, this_veh_tp, valid_path_matrix)
    raw_action = action_l[0]

    # 5.3 校验动作合法性（避开障碍/边界，复用Controller的check_action逻辑）
    # valid_action = controller.check_action(
    #     this_veh_cp=this_veh_cp,
    #     veh_tp=this_veh_tp,
    #     valid_path_matrix=valid_path_matrix,
    #     action=raw_action,
    #     this_veh=test_veh_name
    # )

    # ===================== 6. 输出测试结果 =====================
    action_map = {0: "上", 1: "右", 2: "下", 3: "左"}
    print(f"\n📊 测试结果：")
    print(f"原始预测动作：{raw_action}（{action_map[raw_action]}）")
    # print(f"合法性校验后动作：{valid_action}（{action_map[valid_action]}）")
    print(f"单次推理耗时：{infer_time:.6f} 秒")

    # ===================== 7. 可选：多步动作预测（模拟AGV移动） =====================
    print("\n🚗 多步动作预测（模拟AGV移动）：")
    current_pos = this_veh_cp.copy()
    max_steps = 50  # 最大预测步数
    start_time = time.time()
    for step in range(max_steps):
        # 生成当前步的观测（需更新current_pos到all_info）
        all_info[1][1] = current_pos  # 更新AGV当前位置
        obs, _, _, valid_path_matrix = create_state(all_info, test_veh_name)

        # 动作预测+合法性校验
        action_l, _ = agent.choose_action_test(obs, this_veh_cp, this_veh_tp, valid_path_matrix)
        raw_action = action_l[0]
        # valid_action = controller.check_action(current_pos, this_veh_tp, valid_path_matrix, raw_action, test_veh_name)
        valid_action = raw_action
        # 模拟AGV移动
        if valid_action == 0:  # 上
            current_pos[1] -= 1
        elif valid_action == 1:  # 右
            current_pos[0] += 1
        elif valid_action == 2:  # 下
            current_pos[1] += 1
        elif valid_action == 3:  # 左
            current_pos[0] -= 1

        # 输出步数信息
        print(
            f"第{step + 1}步 | 动作：{action_map[valid_action]} | 当前位置：{current_pos} | 距离目标：{np.linalg.norm(np.array(current_pos) - np.array(this_veh_tp)):.2f}")

        # 到达目标则停止
        if current_pos == this_veh_tp:
            end_time = time.time()
            print(f"✅ 第{step + 1}步到达目标位置！耗时{end_time - start_time:.2f}s")
            break


def run_dqn_model_test():
    """
    支持任务列表依次执行的测试流程
    """
    # ===================== 1. 初始化测试场景 =====================
    ss_x_width, ss_y_width, ss_x_num, ss_y_num, ps_num = 3, 2, 3, 3, 2
    layout = Layout(storage_station_x_width=ss_x_width, storage_station_y_width=ss_y_width,
                    storage_station_x_num=ss_x_num, storage_station_y_num=ss_y_num,
                    picking_station_number=ps_num)

    explorer_num = 1
    explorer_group = []
    for i in range(explorer_num):
        veh_name = "veh" + str(i + 1)
        explorer = Explorer(layout, veh_name=veh_name, icon_name=veh_name)
        explorer_group.append(explorer)

    # ===================== 2. 加载模型 =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net_path = 'algorithm/DQN_structure/323321/train/run1/RMFS_DQN_policy_net.pt'
    target_net_path = 'algorithm/DQN_structure/323321/train/run1/RMFS_DQN_target_net.pt'

    policy_net = torch.load(policy_net_path, map_location=device)
    target_net = torch.load(target_net_path, map_location=device)
    policy_net.eval()
    target_net.eval()
    agent = Agent(policy_net, target_net)

    # ===================== 3. 定义任务列表 =====================
    # 格式: (初始坐标 [x,y], 目标坐标 [x,y], 是否载货 0/1)
    task_queue = [
        ([1, 1], [5, 5], 0),
        ([5, 5], [10, 12], 1),
        ([10, 12], [2, 8], 1),
        ([2, 8], [1, 1], 0)
    ]

    test_veh_name = "veh1"
    action_map = {0: "上", 1: "右", 2: "下", 3: "左"}

    print(f"🚀 开始执行任务队列，共 {len(task_queue)} 个任务")

    # ===================== 4. 依次执行任务 =====================
    for task_idx, (start_pos, target_pos, is_loaded) in enumerate(task_queue):
        print(f"\n--- 任务 {task_idx + 1} ---")
        print(f"起点: {start_pos} | 终点: {target_pos} | 载货: {is_loaded}")

        current_pos = list(start_pos)
        step_count = 0
        max_steps = 100  # 单个任务最大步数，防止死循环

        while current_pos != target_pos and step_count < max_steps:
            # 构造当前环境信息
            # all_info 格式：[layout_matrix, [veh_name, cur_pos, tar_pos, loaded]]
            all_info = [layout.layout, [test_veh_name, current_pos, target_pos, is_loaded]]

            # 1. 生成观测
            obs, this_veh_cp, this_veh_tp, valid_path_matrix = create_state(all_info, test_veh_name)

            # 2. 预测动作
            action_l, _ = agent.choose_action_test(obs, this_veh_cp, this_veh_tp, valid_path_matrix)
            valid_action = action_l[0]

            # 3. 更新位置 (模拟移动)
            if valid_action == 0:  # 上 (y减小)
                current_pos[1] -= 1
            elif valid_action == 1:  # 右 (x增大)
                current_pos[0] += 1
            elif valid_action == 2:  # 下 (y增大)
                current_pos[1] += 1
            elif valid_action == 3:  # 左 (x减小)
                current_pos[0] -= 1

            step_count += 1

            # 打印实时进度
            dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            print(f"步数: {step_count} | 位置: {current_pos} | 动作: {action_map[valid_action]} | 距离目标: {dist:.2f}")

        if current_pos == target_pos:
            print(f"✅ 任务 {task_idx + 1} 完成！总计 {step_count} 步。")
        else:
            print(f"❌ 任务 {task_idx + 1} 失败：达到最大步数未到达。")

    print("\n✨ 所有任务执行完毕！")


def create_state(all_info, this_veh):
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
        create_path_matrix(layout, veh_loaded, current_place, target_place, occupied_place)
    current_position_matrix, target_position_matrix, occupied_position_matrix \
        = create_position_matrix(layout, current_place, target_place, occupied_place, occupied_target)

    state = np.array((current_position_matrix, target_position_matrix, valid_path_matrix))
    """neural network uses state to make decision"""
    """astar algorithm uses current_place, target_place, valid_path_matrix to make decision"""
    return state, current_place, target_place, valid_path_matrix


def create_path_matrix(layout, veh_loaded, current_place, target_place, occupied_place):
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
    basic_matrix = create_basic_matrix(layout)
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


def create_basic_matrix(layout):
    basic_matrix, basic_matrix_one_line = [], []
    for map_one_line in layout:
        for one_cell in map_one_line:
            basic_matrix_one_line.append(0.)
        basic_matrix.append(basic_matrix_one_line)
        basic_matrix_one_line = []
    return basic_matrix


def create_position_matrix(layout, current_place, target_place, occupied_place, occupied_target):
    basic_matrix = create_basic_matrix(layout)
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


if __name__ == '__main__':
    run_dqn_model_test()
