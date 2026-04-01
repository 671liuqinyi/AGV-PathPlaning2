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
from multiAGVscene.Layout import Layout
from utils.astar import FindPathAstar
from utils.task_generate import validate_layout_list

# Keep these keys aligned with mappo.py for same-task benchmarking
LAYOUT_KEY = "layout_list_50"
TASKS_KEY = "tasks_50_generated"
MAX_STEPS_PER_TASK = 300

ACTION_TO_OFFSET = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}


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


def is_valid_next(valid_path_matrix, next_pos):
    map_h = len(valid_path_matrix)
    map_w = len(valid_path_matrix[0])
    x, y = next_pos
    if x < 1 or y < 1 or x > map_w or y > map_h:
        return False
    return valid_path_matrix[y - 1][x - 1] != 0


def step_position(current_pos, action_str):
    dx, dy = ACTION_TO_OFFSET.get(action_str, (0, 0))
    return [current_pos[0] + dx, current_pos[1] + dy]


def choose_astar_action(valid_path_matrix, current_pos, target_pos):
    start_pos_zero = (current_pos[0] - 1, current_pos[1] - 1)
    target_pos_zero = (target_pos[0] - 1, target_pos[1] - 1)

    t0 = time.time()
    path_finder = FindPathAstar(valid_path_matrix, start_pos_zero, target_pos_zero)
    find_target, _, _, action_list = path_finder.run_astar_method()
    decision_time = time.time() - t0

    if not find_target or not action_list:
        return None, decision_time
    return action_list[0], decision_time


def run_astar_test(layout_matrix, task_list, max_steps_per_task=120):
    layout_matrix = validate_layout_list(layout_matrix)
    task_list = validate_tasks(task_list)

    layout = Layout(layout_list=layout_matrix, task_list=[])
    state_manager = StateManager()

    total_decision_time = 0.0
    total_distance = 0.0
    total_collisions = 0
    total_steps = 0
    success_count = 0

    print("A* test started")
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
            _, _, _, valid_path_matrix = state_manager.create_state(all_info, veh_name, obs_clip=True)

            action_str, decision_time = choose_astar_action(valid_path_matrix, current, target)
            task_decision_time += decision_time

            if action_str is None:
                task_collisions += 1
                total_steps += 1
                break

            next_pos = step_position(current, action_str)
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

    print("\n=== A* Test Metrics ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    layout_matrix, task_list = load_layout_and_tasks_from_json(LAYOUT_KEY, TASKS_KEY)
    run_astar_test(layout_matrix=layout_matrix, task_list=task_list, max_steps_per_task=MAX_STEPS_PER_TASK)


if __name__ == "__main__":
    main()
