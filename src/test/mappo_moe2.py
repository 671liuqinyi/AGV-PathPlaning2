import json
import os
import sys
import time

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_JSON_PATH = os.path.join(PROJECT_ROOT, "data.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.Manager.StateManager import StateManager
from algorithm.MAPPO_MOE2_structure.features import extract_expert_features
from multiAGVscene.Layout import Layout
from utils.task_generate import validate_layout_list


ACTION_TO_OFFSET = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
MAX_STEPS_PER_TASK = 200
MODEL_DIR = None

LAYOUT_KEY = "layout_list_50"
TASKS_KEY = "tasks_50_generated"


def resolve_model_paths(base_dir=None):
    actor_candidates = ["RMFS_MAPPO_MOE2_actor_net.pt", "RMFS_MAPPO_MOE2_actor_net_auto.pt"]
    critic_candidates = ["RMFS_MAPPO_MOE2_critic_net.pt", "RMFS_MAPPO_MOE2_critic_net_auto.pt"]

    mappo_root = os.path.join(SRC_DIR, "algorithm", "MAPPO_MOE2_structure")

    if base_dir is None:
        # preferred = os.path.join(mappo_root, "323321")
        preferred = os.path.join(mappo_root, "823632")
        if os.path.isdir(preferred):
            base_dir = preferred
        else:
            dirs = [
                os.path.join(mappo_root, d)
                for d in os.listdir(mappo_root)
                if os.path.isdir(os.path.join(mappo_root, d))
            ]
            dirs = [d for d in dirs if os.path.isdir(os.path.join(d, "train")) or os.path.isdir(os.path.join(d, "test"))]
            if not dirs:
                raise FileNotFoundError("No MAPPO+MoE2 model folder found. Train with mappo_moe2.py first.")
            dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            base_dir = dirs[0]
    else:
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(SRC_DIR, base_dir)
        base_dir = os.path.abspath(base_dir)

    for actor_name, critic_name in zip(actor_candidates, critic_candidates):
        actor_path = os.path.join(base_dir, actor_name)
        critic_path = os.path.join(base_dir, critic_name)
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            return actor_path, critic_path

    test_dir = os.path.join(base_dir, "test")
    for actor_name, critic_name in zip(actor_candidates, critic_candidates):
        test_actor = os.path.join(test_dir, actor_name)
        test_critic = os.path.join(test_dir, critic_name)
        if os.path.exists(test_actor) and os.path.exists(test_critic):
            return test_actor, test_critic

    train_dir = os.path.join(base_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train folder not found: {train_dir}")

    run_dirs = []
    for dir_name in os.listdir(train_dir):
        if dir_name.startswith("run") and dir_name[3:].isdigit():
            run_dirs.append((int(dir_name[3:]), os.path.join(train_dir, dir_name)))
    run_dirs.sort(key=lambda item: item[0], reverse=True)

    for _, run_dir in run_dirs:
        for actor_name, critic_name in zip(actor_candidates, critic_candidates):
            actor_path = os.path.join(run_dir, actor_name)
            critic_path = os.path.join(run_dir, critic_name)
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                return actor_path, critic_path

    raise FileNotFoundError(f"No usable MAPPO+MoE2 model files found under: {base_dir}")


def load_layout_and_tasks_from_json(layout_key, tasks_key):
    if not os.path.exists(DATA_JSON_PATH):
        raise FileNotFoundError(f"data.json not found: {DATA_JSON_PATH}")

    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if layout_key is None:
        layout_key = payload.get("_latest_layout_key")
    if tasks_key is None:
        tasks_key = payload.get("_latest_tasks_key")

    if layout_key is None or tasks_key is None:
        raise KeyError("Missing layout/tasks key and no _latest_* metadata in data.json")
    if layout_key not in payload:
        raise KeyError(f"layout key '{layout_key}' not found in data.json")
    if tasks_key not in payload:
        raise KeyError(f"tasks key '{tasks_key}' not found in data.json")

    layout_matrix = payload[layout_key]
    task_list = payload[tasks_key]
    print(f"Using data.json keys: layout='{layout_key}', tasks='{tasks_key}'")
    return layout_matrix, task_list


def create_action_mask(valid_path_matrix, current_place):
    action_mask = np.zeros(4, dtype=np.float32)
    map_h = len(valid_path_matrix)
    map_w = len(valid_path_matrix[0])

    for action, (dx, dy) in ACTION_TO_OFFSET.items():
        nx = current_place[0] + dx
        ny = current_place[1] + dy
        if 1 <= nx <= map_w and 1 <= ny <= map_h and valid_path_matrix[ny - 1][nx - 1] != 0:
            action_mask[action] = 1.0

    if action_mask.sum() == 0:
        action_mask[:] = 1.0
    return action_mask


def step_position(current_pos, action):
    dx, dy = ACTION_TO_OFFSET.get(action, (0, 0))
    return [current_pos[0] + dx, current_pos[1] + dy]


def is_valid_next(valid_path_matrix, next_pos):
    map_h = len(valid_path_matrix)
    map_w = len(valid_path_matrix[0])
    x, y = next_pos
    if x < 1 or y < 1 or x > map_w or y > map_h:
        return False
    return valid_path_matrix[y - 1][x - 1] != 0


def load_agent(model_dir=None):
    import torch
    from algorithm.MAPPO_MOE2_structure.MAPPO_MOE2 import MAPPOMoE2Agent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path, critic_path = resolve_model_paths(model_dir)
    actor_net = torch.load(actor_path, map_location=device)
    critic_net = torch.load(critic_path, map_location=device)
    actor_net.eval()
    critic_net.eval()
    return MAPPOMoE2Agent(actor_net, critic_net, device), actor_path, critic_path


def validate_tasks(task_list):
    if not isinstance(task_list, list) or not task_list:
        raise ValueError("tasks must be a non-empty list.")

    for idx, task in enumerate(task_list, start=1):
        if not isinstance(task, dict) or len(task) != 1:
            raise ValueError(f"task[{idx}] must be a dict with exactly one vehicle key.")

        veh_name = next(iter(task.keys()))
        info = task[veh_name]
        if not isinstance(info, dict):
            raise ValueError(f"task[{idx}]['{veh_name}'] must be a dict.")

        for key in ("current", "target", "loaded"):
            if key not in info:
                raise ValueError(f"task[{idx}]['{veh_name}'] missing key: {key}")

        current = info["current"]
        target = info["target"]
        loaded = info["loaded"]
        if not isinstance(current, list) or len(current) != 2 or not isinstance(target, list) or len(target) != 2:
            raise ValueError(f"task[{idx}] current/target must be [x, y].")
        if loaded not in (0, 1, False, True):
            raise ValueError(f"task[{idx}] loaded must be 0/1.")

    return task_list


def run_mappo_moe2_test(layout_matrix, task_list, max_steps_per_task=120, model_dir=None):
    layout_matrix = validate_layout_list(layout_matrix)
    task_list = validate_tasks(task_list)

    layout = Layout(layout_list=layout_matrix, task_list=[])
    state_manager = StateManager()
    agent, actor_path, critic_path = load_agent(model_dir)
    map_h, map_w = len(layout_matrix), len(layout_matrix[0])

    total_decision_time = 0.0
    total_distance = 0.0
    total_collisions = 0
    total_steps = 0
    success_count = 0

    print("MAPPO+MoE2 model loaded")
    print("actor:", actor_path)
    print("critic:", critic_path)
    print(f"task_count={len(task_list)}")

    for idx, task in enumerate(task_list, start=1):
        veh_name = next(iter(task.keys()))
        info = task[veh_name]
        current = list(info["current"])
        target = list(info["target"])
        loaded = int(info["loaded"])

        task_distance = 0.0
        task_collisions = 0
        task_decision_time = 0.0
        reached = False

        for _ in range(max_steps_per_task):
            all_info = [layout.layout, [veh_name, list(current), list(target), loaded]]
            obs, cp, _, valid_path_matrix = state_manager.create_state(all_info, veh_name, obs_clip=True)
            obs = np.array(obs, dtype=np.float32)
            action_mask = create_action_mask(valid_path_matrix, cp)
            expert_features = extract_expert_features(all_info, veh_name, valid_path_matrix, map_w, map_h)

            t0 = time.time()
            action, _, _, _ = agent.choose_action(
                obs,
                expert_features=expert_features,
                action_mask=action_mask,
                deterministic=True,
            )
            task_decision_time += time.time() - t0

            next_pos = step_position(current, action)
            if is_valid_next(valid_path_matrix, next_pos):
                task_distance += float(np.linalg.norm(np.array(next_pos) - np.array(current)))
                current = next_pos
            else:
                task_collisions += 1

            total_steps += 1
            if current == target:
                reached = True
                break

        total_decision_time += task_decision_time
        total_distance += task_distance
        total_collisions += task_collisions
        success_count += 1 if reached else 0

        print(
            f"task={idx:03d} reached={reached} "
            f"decision_time={task_decision_time:.6f}s distance={task_distance:.2f} collisions={task_collisions}"
        )

    avg_decision_time = total_decision_time / max(total_steps, 1)
    result = {
        "task_count": len(task_list),
        "success_count": success_count,
        "total_decision_time_s": round(total_decision_time, 6),
        "avg_decision_time_per_step_s": round(avg_decision_time, 8),
        "total_movement_distance": round(total_distance, 2),
        "total_collision_count": int(total_collisions),
        "total_steps": int(total_steps),
    }

    print("\n=== MAPPO+MoE2 Test Metrics ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    layout_matrix, task_list = load_layout_and_tasks_from_json(LAYOUT_KEY, TASKS_KEY)
    run_mappo_moe2_test(
        layout_matrix=layout_matrix,
        task_list=task_list,
        max_steps_per_task=MAX_STEPS_PER_TASK,
        model_dir=MODEL_DIR,
    )


if __name__ == "__main__":
    main()
