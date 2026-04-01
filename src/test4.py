import argparse
import copy
import os
import time

import numpy as np
import torch

from algorithm.MAPPO_structure.MAPPO import MAPPOAgent
from algorithm.Manager.StateManager import StateManager
from multiAGVscene.Explorer import Explorer
from multiAGVscene.Layout import Layout
from multiAGVscene.Scene import Scene

ACTION_MAP = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_TO_OFFSET = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def resolve_model_paths(base_dir=None):
    """
    模型路径查找顺序：
    1) 指定目录/test 下正式模型
    2) 指定目录/test 下 auto 模型
    3) 指定目录/train 最新 runN 下正式模型
    4) 指定目录/train 最新 runN 下 auto 模型
    """
    actor_candidates = ["RMFS_MAPPO_actor_net.pt", "RMFS_MAPPO_actor_net_auto.pt"]
    critic_candidates = ["RMFS_MAPPO_critic_net.pt", "RMFS_MAPPO_critic_net_auto.pt"]

    src_dir = os.path.dirname(os.path.abspath(__file__))
    mappo_root = os.path.join(src_dir, "algorithm", "MAPPO_structure")

    if base_dir is None:
        preferred = os.path.join(mappo_root, "423421")
        if os.path.isdir(preferred):
            base_dir = preferred
        else:
            dirs = [os.path.join(mappo_root, d) for d in os.listdir(mappo_root) if os.path.isdir(os.path.join(mappo_root, d))]
            dirs = [d for d in dirs if os.path.isdir(os.path.join(d, "train")) or os.path.isdir(os.path.join(d, "test"))]
            if not dirs:
                raise FileNotFoundError("未找到 MAPPO 模型目录，请先运行 main4.py 训练。")
            dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            base_dir = dirs[0]
    else:
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(src_dir, base_dir)
        base_dir = os.path.abspath(base_dir)

    test_dir = os.path.join(base_dir, "test")
    for actor_name, critic_name in zip(actor_candidates, critic_candidates):
        test_actor = os.path.join(test_dir, actor_name)
        test_critic = os.path.join(test_dir, critic_name)
        if os.path.exists(test_actor) and os.path.exists(test_critic):
            return test_actor, test_critic

    train_dir = os.path.join(base_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"未找到 train 目录: {train_dir}")

    run_dirs = []
    for dir_name in os.listdir(train_dir):
        if not dir_name.startswith("run"):
            continue
        num_str = dir_name.replace("run", "", 1)
        if num_str.isdigit():
            run_dirs.append((int(num_str), os.path.join(train_dir, dir_name)))

    run_dirs.sort(key=lambda x: x[0], reverse=True)

    for _, run_dir in run_dirs:
        for actor_name, critic_name in zip(actor_candidates, critic_candidates):
            actor_path = os.path.join(run_dir, actor_name)
            critic_path = os.path.join(run_dir, critic_name)
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                return actor_path, critic_path

    raise FileNotFoundError(f"未找到可用的 MAPPO 模型文件，请先训练。搜索目录: {base_dir}")


def create_action_mask(valid_path_matrix, current_place):
    action_mask = np.zeros(4, dtype=np.float32)
    map_ydim = len(valid_path_matrix)
    map_xdim = len(valid_path_matrix[0])

    for action in range(4):
        dx, dy = ACTION_TO_OFFSET[action]
        next_x = current_place[0] + dx
        next_y = current_place[1] + dy

        if 1 <= next_x <= map_xdim and 1 <= next_y <= map_ydim:
            if valid_path_matrix[next_y - 1][next_x - 1] != 0:
                action_mask[action] = 1.0

    if action_mask.sum() == 0:
        action_mask[:] = 1.0

    return action_mask


def step_position(current_pos, action):
    new_pos = list(current_pos)
    if action in ACTION_TO_OFFSET:
        dx, dy = ACTION_TO_OFFSET[action]
        new_pos[0] += dx
        new_pos[1] += dy
    return new_pos


def build_scene(explorer_num):
    ss_x_width, ss_y_width, ss_x_num, ss_y_num, ps_num = 4, 2, 3, 4, 2
    layout = Layout(
        storage_station_x_width=ss_x_width,
        storage_station_y_width=ss_y_width,
        storage_station_x_num=ss_x_num,
        storage_station_y_num=ss_y_num,
        picking_station_number=ps_num,
    )

    explorer_group = []
    for idx in range(explorer_num):
        veh_name = f"veh{idx + 1}"
        explorer_group.append(Explorer(layout, veh_name=veh_name, icon_name=veh_name))

    _ = Scene(layout, explorer_group)
    return layout, explorer_group


def load_agent(model_base_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path, critic_path = resolve_model_paths(model_base_dir)

    actor_net = torch.load(actor_path, map_location=device)
    critic_net = torch.load(critic_path, map_location=device)
    actor_net.eval()
    critic_net.eval()

    agent = MAPPOAgent(actor_net, critic_net, device)
    return agent, actor_path, critic_path


def build_all_info(layout, vehicle_states):
    all_info = [layout.layout]
    for veh_name in sorted(vehicle_states.keys()):
        v = vehicle_states[veh_name]
        all_info.append([veh_name, list(v["current"]), list(v["target"]), int(v["loaded"])])
    return all_info


def find_road_cells(layout):
    road_cells = []
    for y, row in enumerate(layout.layout, start=1):
        for x, cell in enumerate(row, start=1):
            if cell == 0:
                road_cells.append([x, y])
    return road_cells


def run_single_vehicle_path_planning_test(max_steps=120, model_base_dir=None):
    """
    单车路径规划测试：单车从起点移动到终点。
    """
    layout, _ = build_scene(explorer_num=1)
    agent, actor_path, critic_path = load_agent(model_base_dir)
    state_manager = StateManager()

    vehicle_state = {"veh1": {"current": [1, 1], "target": [10, 12], "loaded": 1}}

    print("=== 单车路径规划测试 ===")
    print("actor:", actor_path)
    print("critic:", critic_path)
    print("任务: veh1", vehicle_state["veh1"])

    start_time = time.time()
    for step in range(1, max_steps + 1):
        all_info = build_all_info(layout, vehicle_state)
        obs, cp, _, valid_path_matrix = state_manager.create_state(all_info, "veh1", obs_clip=True)
        obs = np.array(obs, dtype=np.float32)

        action_mask = create_action_mask(valid_path_matrix, cp)
        t0 = time.time()
        action, _ = agent.choose_action(obs, action_mask=action_mask, deterministic=True)
        infer_time = time.time() - t0

        vehicle_state["veh1"]["current"] = step_position(vehicle_state["veh1"]["current"], action)
        dist = np.linalg.norm(np.array(vehicle_state["veh1"]["current"]) - np.array(vehicle_state["veh1"]["target"]))

        print(
            f"step={step:03d} | action={ACTION_MAP[action]} | pos={vehicle_state['veh1']['current']} | "
            f"dist={dist:.2f} | infer={infer_time:.6f}s"
        )

        if vehicle_state["veh1"]["current"] == vehicle_state["veh1"]["target"]:
            print(f"单车测试成功: 到达目标，总耗时 {time.time() - start_time:.2f}s")
            return True

    print(f"单车测试结束: 达到最大步数 {max_steps}，未到达目标")
    return False


def run_multi_vehicle_path_planning_test(max_steps=200, model_base_dir=None):
    """
    多车路径规划测试：多车同时规划，按固定优先级顺序逐车决策。
    """
    explorer_num = 2
    layout, _ = build_scene(explorer_num=explorer_num)
    agent, actor_path, critic_path = load_agent(model_base_dir)
    state_manager = StateManager()

    road_cells = find_road_cells(layout)
    if len(road_cells) < explorer_num * 2:
        raise RuntimeError("道路可用栅格数量不足，无法构造多车测试任务。")

    vehicle_states = {
        "veh1": {"current": list(road_cells[0]), "target": list(road_cells[-1]), "loaded": 1},
        "veh2": {"current": list(road_cells[1]), "target": list(road_cells[-2]), "loaded": 0},
    }

    print("\n=== 多车路径规划测试 ===")
    print("actor:", actor_path)
    print("critic:", critic_path)
    for veh_name in sorted(vehicle_states.keys()):
        print(f"任务: {veh_name} {vehicle_states[veh_name]}")

    start_time = time.time()
    no_progress_steps = 0

    for step in range(1, max_steps + 1):
        pre_positions = {k: list(v["current"]) for k, v in vehicle_states.items()}
        planned_states = copy.deepcopy(vehicle_states)
        reserved_positions = set()

        for veh_name in sorted(vehicle_states.keys()):
            all_info = build_all_info(layout, planned_states)
            obs, cp, _, valid_path_matrix = state_manager.create_state(all_info, veh_name, obs_clip=True)
            obs = np.array(obs, dtype=np.float32)

            action_mask = create_action_mask(valid_path_matrix, cp)
            action, _ = agent.choose_action(obs, action_mask=action_mask, deterministic=True)

            current_pos = list(planned_states[veh_name]["current"])
            next_pos = step_position(current_pos, action)
            next_key = tuple(next_pos)

            if next_key in reserved_positions:
                next_pos = current_pos
                action_name = "HOLD"
            else:
                action_name = ACTION_MAP[action]

            planned_states[veh_name]["current"] = next_pos
            reserved_positions.add(tuple(next_pos))

            dist = np.linalg.norm(np.array(next_pos) - np.array(planned_states[veh_name]["target"]))
            print(f"step={step:03d} | {veh_name} | action={action_name:>5} | pos={next_pos} | dist={dist:.2f}")

        vehicle_states = planned_states
        moved = any(vehicle_states[name]["current"] != pre_positions[name] for name in vehicle_states.keys())
        no_progress_steps = 0 if moved else no_progress_steps + 1

        finished = [name for name, v in vehicle_states.items() if v["current"] == v["target"]]
        if len(finished) == len(vehicle_states):
            print(f"多车测试成功: 全部车辆到达目标，总耗时 {time.time() - start_time:.2f}s")
            return True

        if no_progress_steps >= 20:
            print("多车测试提前停止: 连续 20 步无进展，疑似死锁。")
            break

    unfinished = [name for name, v in vehicle_states.items() if v["current"] != v["target"]]
    print(f"多车测试结束: 达到最大步数 {max_steps} 或提前停止，未完成车辆: {unfinished}")
    return False


def main():
    parser = argparse.ArgumentParser(description="MAPPO 路径规划测试脚本（单车/多车）")
    parser.add_argument(
        "--mode",
        type=str,
        default="multi",
        choices=["single", "multi", "both"],
        help="single: 单车测试, multi: 多车测试, both: 两者都跑",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="algorithm/MAPPO_structure/423421",
        help="可选：指定 MAPPO 模型根目录（例如 algorithm/MAPPO_structure/423421）",
    )
    parser.add_argument("--single-steps", type=int, default=120, help="单车测试最大步数")
    parser.add_argument("--multi-steps", type=int, default=200, help="多车测试最大步数")
    args = parser.parse_args()

    if args.mode in ["single", "both"]:
        run_single_vehicle_path_planning_test(max_steps=args.single_steps, model_base_dir=args.model_dir)

    if args.mode in ["multi", "both"]:
        run_multi_vehicle_path_planning_test(max_steps=args.multi_steps, model_base_dir=args.model_dir)


if __name__ == "__main__":
    main()
