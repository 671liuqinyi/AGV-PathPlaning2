import json
import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_JSON_PATH = os.path.join(PROJECT_ROOT, "data.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.DQN_structure.DQN import Agent
from multiAGVscene.Layout import Layout
from utils.task_generate import validate_layout_list

# Keep same task source convention as mappo.py / Astar.py
LAYOUT_KEY = "layout_list_30"
TASKS_KEY = "tasks_30_generated"
MAX_STEPS_PER_TASK = 120
DQN_MODEL_DIR = "algorithm/DQN_structure/323321"  # e.g. "algorithm/DQN_structure/323321"

ACTION_TO_OFFSET = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def load_layout_and_tasks_from_json(layout_key, tasks_key):
    if not os.path.exists(DATA_JSON_PATH):
        raise FileNotFoundError(f"data.json not found: {DATA_JSON_PATH}")

    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if layout_key not in payload:
        latest_layout_key = payload.get("_latest_layout_key")
        if latest_layout_key in payload:
            layout_key = latest_layout_key
        else:
            raise KeyError(f"layout key '{layout_key}' not found in data.json")

    if tasks_key not in payload:
        latest_tasks_key = payload.get("_latest_tasks_key")
        if latest_tasks_key in payload:
            tasks_key = latest_tasks_key
        else:
            raise KeyError(f"tasks key '{tasks_key}' not found in data.json")

    layout_matrix = payload[layout_key]
    task_list = payload[tasks_key]
    print(f"Using data.json keys: layout='{layout_key}', tasks='{tasks_key}'")
    return layout_matrix, task_list


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


def resolve_dqn_model_path_candidates(base_dir=None):
    policy_candidates = ["RMFS_DQN_policy_net.pt", "RMFS_DQN_policy_net_auto.pt"]
    target_candidates = ["RMFS_DQN_target_net.pt", "RMFS_DQN_target_net_auto.pt"]

    dqn_root = os.path.join(SRC_DIR, "algorithm", "DQN_structure")

    if base_dir is None:
        preferred = os.path.join(dqn_root, "323321")
        if os.path.isdir(preferred):
            base_dir = preferred
        else:
            dirs = [os.path.join(dqn_root, d) for d in os.listdir(dqn_root) if os.path.isdir(os.path.join(dqn_root, d))]
            dirs = [d for d in dirs if os.path.isdir(os.path.join(d, "train")) or os.path.isdir(os.path.join(d, "test"))]
            if not dirs:
                raise FileNotFoundError("No DQN model folder found. Train with main.py first.")
            dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            base_dir = dirs[0]
    else:
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(SRC_DIR, base_dir)
        base_dir = os.path.abspath(base_dir)

    candidates = []

    # Prefer train/runN checkpoints first (usually match latest training config).
    train_dir = os.path.join(base_dir, "train")
    if os.path.isdir(train_dir):
        run_dirs = []
        for dir_name in os.listdir(train_dir):
            if dir_name.startswith("run") and dir_name[3:].isdigit():
                run_dirs.append((int(dir_name[3:]), os.path.join(train_dir, dir_name)))
        run_dirs.sort(key=lambda x: x[0], reverse=True)

        for _, run_dir in run_dirs:
            for policy_name, target_name in zip(policy_candidates, target_candidates):
                p_path = os.path.join(run_dir, policy_name)
                t_path = os.path.join(run_dir, target_name)
                if os.path.exists(p_path) and os.path.exists(t_path):
                    candidates.append((p_path, t_path))

        # Then train root folder.
        for policy_name, target_name in zip(policy_candidates, target_candidates):
            p_path = os.path.join(train_dir, policy_name)
            t_path = os.path.join(train_dir, target_name)
            if os.path.exists(p_path) and os.path.exists(t_path):
                candidates.append((p_path, t_path))

    # Fallback to test folder.
    test_dir = os.path.join(base_dir, "test")
    for policy_name, target_name in zip(policy_candidates, target_candidates):
        p_path = os.path.join(test_dir, policy_name)
        t_path = os.path.join(test_dir, target_name)
        if os.path.exists(p_path) and os.path.exists(t_path):
            candidates.append((p_path, t_path))

    # Fallback to base folder.
    for policy_name, target_name in zip(policy_candidates, target_candidates):
        p_path = os.path.join(base_dir, policy_name)
        t_path = os.path.join(base_dir, target_name)
        if os.path.exists(p_path) and os.path.exists(t_path):
            candidates.append((p_path, t_path))

    if not candidates:
        raise FileNotFoundError(f"No usable DQN model files found under: {base_dir}")

    # Deduplicate while preserving order.
    unique_candidates = []
    seen = set()
    for p_path, t_path in candidates:
        key = (os.path.abspath(p_path), os.path.abspath(t_path))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((p_path, t_path))
    return unique_candidates


def infer_model_pooled_area(policy_network):
    fc3 = getattr(policy_network, "fc3", None)
    if fc3 is None:
        return None

    in_features = getattr(fc3, "in_features", None)
    if not isinstance(in_features, int):
        return None
    if in_features <= 0 or in_features % 256 != 0:
        return None
    return in_features // 256


def load_dqn_agent(model_dir=None, layout_h=None, layout_w=None):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_root = os.path.join(SRC_DIR, "algorithm", "DQN_structure")

    base_dirs = []
    if model_dir is not None:
        preferred_base = model_dir
        if not os.path.isabs(preferred_base):
            preferred_base = os.path.join(SRC_DIR, preferred_base)
        preferred_base = os.path.abspath(preferred_base)
        base_dirs.append(preferred_base)
    else:
        base_dirs.append(None)

    # Auto-scan other DQN model folders as fallback.
    if os.path.isdir(dqn_root):
        for name in os.listdir(dqn_root):
            one_dir = os.path.join(dqn_root, name)
            if not os.path.isdir(one_dir):
                continue
            if one_dir in base_dirs:
                continue
            base_dirs.append(one_dir)

    candidates = []
    for base in base_dirs:
        try:
            candidates.extend(resolve_dqn_model_path_candidates(base))
        except FileNotFoundError:
            continue
    if not candidates:
        raise FileNotFoundError("No usable DQN model files found in configured directories.")

    pooled_current = None
    if layout_h is not None and layout_w is not None:
        pooled_current = (int(layout_h) // 4) * (int(layout_w) // 4)

    fallback = None
    checked = []
    for policy_path, target_path in candidates:
        try:
            policy_net = torch.load(policy_path, map_location=device)
            target_net = torch.load(target_path, map_location=device)
            policy_net.eval()
            target_net.eval()
        except Exception as ex:
            checked.append(f"{policy_path} -> load_failed: {ex}")
            continue

        pooled_expected = infer_model_pooled_area(policy_net)
        checked.append(f"{policy_path} -> pooled_expected={pooled_expected}")

        if fallback is None:
            fallback = (policy_net, target_net, policy_path, target_path, pooled_expected)

        if pooled_current is None:
            agent = Agent(policy_net, target_net)
            return agent, policy_path, target_path, pooled_expected

        if pooled_expected is not None and pooled_expected == pooled_current:
            agent = Agent(policy_net, target_net)
            return agent, policy_path, target_path, pooled_expected

    if fallback is None:
        raise RuntimeError("DQN model loading failed for all candidates:\n" + "\n".join(checked))

    if pooled_current is not None:
        debug = "\n".join(checked)
        raise RuntimeError(
            "DQN model input size mismatch (no-resize mode). "
            f"layout pooled area={pooled_current}, but no candidate checkpoint matched.\n"
            "Checked candidates:\n"
            f"{debug}\n"
            "Please switch LAYOUT_KEY/TASKS_KEY or use a matching DQN checkpoint."
        )

    policy_net, target_net, policy_path, target_path, pooled_expected = fallback
    agent = Agent(policy_net, target_net)
    return agent, policy_path, target_path, pooled_expected


def validate_model_layout_compat(policy_network, layout_h, layout_w):
    """
    Keep pure-DQN behavior (no resizing): only check compatibility early.
    DQN Net expects: fc3.in_features = 256 * floor(H/4) * floor(W/4)
    """
    if not hasattr(policy_network, "fc3"):
        return

    in_features = int(policy_network.fc3.in_features)
    if in_features <= 0 or in_features % 256 != 0:
        return

    pooled_expected = in_features // 256
    pooled_current = (layout_h // 4) * (layout_w // 4)

    if pooled_expected != pooled_current:
        raise RuntimeError(
            "DQN model input size mismatch (no-resize mode). "
            f"model expects pooled area={pooled_expected}, but current layout gives {pooled_current} "
            f"(layout={layout_h}x{layout_w}). "
            "Please use matching layout/tasks for this model, or switch to a matching DQN checkpoint."
        )


def create_state(all_info, this_veh):
    layout = all_info[0]
    occupied_place = []
    occupied_target = []
    current_place = 0
    target_place = 0
    veh_loaded = False

    for i in range(1, len(all_info)):
        one_veh = all_info[i]
        veh_name_, current_place_, target_place_, veh_loaded_ = one_veh[0], one_veh[1], one_veh[2], one_veh[3]
        if veh_name_ == this_veh:
            current_place, target_place, veh_loaded = current_place_, target_place_, veh_loaded_
        else:
            occupied_place.append(current_place_)
            occupied_target.append(target_place_)

    valid_path_matrix, _, _ = create_path_matrix(layout, veh_loaded, current_place, target_place, occupied_place)
    current_position_matrix, target_position_matrix, _ = create_position_matrix(layout, current_place, target_place, occupied_place, occupied_target)

    state = np.array((current_position_matrix, target_position_matrix, valid_path_matrix), dtype=np.float32)
    return state, current_place, target_place, valid_path_matrix


def create_path_matrix(layout, veh_loaded, current_place, target_place, occupied_place):
    valid_path = []
    forbidden_path = []

    for map_one_line in layout:
        valid_path_one_line = []
        forbidden_path_one_line = []
        for one_cell in map_one_line:
            if one_cell == 0:
                valid_path_one_line.append(1.0)
                forbidden_path_one_line.append(0.0)
            elif one_cell == 1:
                if veh_loaded == 0:
                    valid_path_one_line.append(1.0)
                    forbidden_path_one_line.append(0.0)
                else:
                    valid_path_one_line.append(0.0)
                    forbidden_path_one_line.append(1.0)
            elif one_cell == 2:
                valid_path_one_line.append(0.0)
                forbidden_path_one_line.append(1.0)
            else:
                raise ValueError("create_path_matrix: wrong matrix value")
        valid_path.append(valid_path_one_line)
        forbidden_path.append(forbidden_path_one_line)

    valid_path_matrix = np.array(valid_path, dtype=np.float32)
    forbidden_path_matrix = np.array(forbidden_path, dtype=np.float32)

    valid_path_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0
    forbidden_path_matrix[current_place[1] - 1][current_place[0] - 1] = 0.0
    valid_path_matrix[target_place[1] - 1][target_place[0] - 1] = 1.0
    forbidden_path_matrix[target_place[1] - 1][target_place[0] - 1] = 0.0

    for o_place in occupied_place:
        valid_path_matrix[o_place[1] - 1][o_place[0] - 1] = 0.0
        forbidden_path_matrix[o_place[1] - 1][o_place[0] - 1] = 1.0

    basic_matrix_array = np.array(create_basic_matrix(layout), dtype=np.float32)
    current_p_x, current_p_y = current_place[0] - 1, current_place[1] - 1
    four_dict = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    if valid_path_matrix[current_p_y][current_p_x] != 0:
        basic_matrix_array[current_p_y][current_p_x] = 1.0
        for one_direction in four_dict:
            pos = [current_p_x + one_direction[0], current_p_y + one_direction[1]]
            if pos[0] < 0 or pos[1] < 0 or pos[0] >= len(valid_path_matrix[0]) or pos[1] >= len(valid_path_matrix):
                continue
            if valid_path_matrix[pos[1]][pos[0]] != 0:
                basic_matrix_array[pos[1]][pos[0]] = 1.0
            else:
                basic_matrix_array[pos[1]][pos[0]] = -1.0

    return valid_path_matrix, forbidden_path_matrix, basic_matrix_array


def create_basic_matrix(layout):
    return [[0.0 for _ in row] for row in layout]


def create_position_matrix(layout, current_place, target_place, occupied_place, occupied_target):
    basic_matrix_array = np.array(create_basic_matrix(layout), dtype=np.float32)

    current_position_matrix = basic_matrix_array.copy()
    current_position_matrix[current_place[1] - 1][current_place[0] - 1] = 1.0

    target_position_matrix = basic_matrix_array.copy()
    target_position_matrix[target_place[1] - 1][target_place[0] - 1] = 1.0

    occupied_position_matrix = basic_matrix_array.copy()
    for occupied_ in occupied_place:
        occupied_position_matrix[occupied_[1] - 1][occupied_[0] - 1] = 1.0

    return current_position_matrix, target_position_matrix, occupied_position_matrix


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


def run_dqn_test(layout_matrix, task_list, max_steps_per_task=120, model_dir=None):
    layout_matrix = validate_layout_list(layout_matrix)
    task_list = validate_tasks(task_list)

    _ = Layout(layout_list=layout_matrix, task_list=[])
    layout_h, layout_w = len(layout_matrix), len(layout_matrix[0])
    pooled_current = (layout_h // 4) * (layout_w // 4)
    agent, policy_path, target_path, pooled_expected = load_dqn_agent(model_dir, layout_h, layout_w)
    validate_model_layout_compat(agent.policy_network, len(layout_matrix), len(layout_matrix[0]))

    total_decision_time = 0.0
    total_distance = 0.0
    total_collisions = 0
    total_steps = 0
    success_count = 0

    print("DQN model loaded")
    print("policy:", policy_path)
    print("target:", target_path)
    print(f"layout={layout_h}x{layout_w}, pooled_current={pooled_current}, pooled_expected={pooled_expected}")
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
            all_info = [layout_matrix, [veh_name, list(current), list(target), loaded]]
            obs, cp, tp, valid_path_matrix = create_state(all_info, veh_name)

            # Follow src/test.py inference style directly: no resizing, no astar fallback.
            action_arr, infer_time = agent.choose_action_test(obs, cp, tp, valid_path_matrix)
            action = int(action_arr[0])
            task_decision_time += float(infer_time)

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

    print("\n=== DQN Test Metrics ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    layout_matrix, task_list = load_layout_and_tasks_from_json(LAYOUT_KEY, TASKS_KEY)
    run_dqn_test(
        layout_matrix=layout_matrix,
        task_list=task_list,
        max_steps_per_task=MAX_STEPS_PER_TASK,
        model_dir=DQN_MODEL_DIR,
    )


if __name__ == "__main__":
    main()
