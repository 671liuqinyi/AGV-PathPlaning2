import argparse
import json
import os
import random
import sys
from collections import deque

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_JSON_PATH = os.path.join(PROJECT_ROOT, "data.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import data

DEFAULT_LAYOUT_KEY = "layout_list_15"
DEFAULT_TASKS_KEY = "tasks_15_generated"


def validate_layout_list(layout_list):
    if (
        not isinstance(layout_list, list)
        or not layout_list
        or not isinstance(layout_list[0], list)
    ):
        raise ValueError("layout_list must be a non-empty 2D list.")

    width = len(layout_list[0])
    for row in layout_list:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("layout_list must be a rectangular 2D list.")
        for cell in row:
            if cell not in (0, 1, 2):
                raise ValueError("layout_list only supports cell values 0, 1, 2.")
    return layout_list


def load_data_json():
    if not os.path.exists(DATA_JSON_PATH):
        return {}
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)


def save_data_json(payload):
    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_layout_from_sources(layout_key=DEFAULT_LAYOUT_KEY):
    """
    Read layout in this order:
    1) data.json[layout_key]
    2) data.py variable with same name (and sync to data.json)
    """
    data_json = load_data_json()
    if layout_key in data_json:
        return validate_layout_list(data_json[layout_key])

    if not hasattr(data, layout_key):
        raise AttributeError(
            f"layout key '{layout_key}' not found in data.json or data.py"
        )

    layout = validate_layout_list(getattr(data, layout_key))
    data_json[layout_key] = layout
    save_data_json(data_json)
    return layout


def append_tasks_to_data_json(tasks_key, tasks):
    data_json = load_data_json()
    if tasks_key not in data_json:
        data_json[tasks_key] = []
    if not isinstance(data_json[tasks_key], list):
        raise ValueError(f"data.json key '{tasks_key}' is not a list, cannot append.")

    data_json[tasks_key].extend(tasks)
    save_data_json(data_json)
    return len(data_json[tasks_key])


def set_latest_keys(layout_key, tasks_key):
    data_json = load_data_json()
    data_json["_latest_layout_key"] = layout_key
    data_json["_latest_tasks_key"] = tasks_key
    save_data_json(data_json)


def extract_cells(layout_list):
    shelf_cells = []
    picking_cells = []
    traversable_for_empty_vehicle = []

    for y, row in enumerate(layout_list, start=1):
        for x, value in enumerate(row, start=1):
            pos = [x, y]
            if value == 0:
                traversable_for_empty_vehicle.append(pos)
            elif value == 1:
                shelf_cells.append(pos)
                traversable_for_empty_vehicle.append(pos)
            elif value == 2:
                picking_cells.append(pos)

    return shelf_cells, picking_cells, traversable_for_empty_vehicle


def is_reachable(layout_list, start, target, loaded):
    h = len(layout_list)
    w = len(layout_list[0])
    sx, sy = start
    tx, ty = target

    def in_bounds(px, py):
        return 1 <= px <= w and 1 <= py <= h

    def passable(px, py):
        if [px, py] == start or [px, py] == target:
            return True
        value = layout_list[py - 1][px - 1]
        if value == 0:
            return True
        if value == 1:
            return loaded == 0
        return False

    if not in_bounds(sx, sy) or not in_bounds(tx, ty):
        return False

    queue = deque([(sx, sy)])
    visited = {(sx, sy)}
    offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while queue:
        x, y = queue.popleft()
        if (x, y) == (tx, ty):
            return True

        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not in_bounds(nx, ny):
                continue
            if not passable(nx, ny):
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))
    return False


def generate_tasks(layout_list, task_count, veh_name="veh1", seed=None):
    """
    Pickup task:
    - start: random traversable cell
    - target: random shelf
    - loaded: 0

    Delivery task:
    - start: random shelf
    - target: random picking station
    - loaded: 1
    """
    if task_count <= 0:
        return []

    if seed is not None:
        random.seed(seed)

    layout = validate_layout_list(layout_list)
    shelf_cells, picking_cells, traversable_for_empty_vehicle = extract_cells(layout)

    if not shelf_cells:
        raise ValueError("layout_list must contain at least one shelf cell (value=1).")
    if not picking_cells:
        raise ValueError(
            "layout_list must contain at least one picking-station cell (value=2)."
        )

    tasks = []
    max_retry_per_task = 200

    for idx in range(task_count):
        task_kind = "pickup" if idx % 2 == 0 else "delivery"
        found = False

        for _ in range(max_retry_per_task):
            if task_kind == "pickup":
                current = random.choice(traversable_for_empty_vehicle)
                target = random.choice(shelf_cells)
                loaded = 0
            else:
                current = random.choice(shelf_cells)
                target = random.choice(picking_cells)
                loaded = 1

            if current == target:
                continue

            if is_reachable(layout, current, target, loaded):
                tasks.append(
                    {
                        veh_name: {
                            "current": list(current),
                            "target": list(target),
                            "loaded": loaded,
                        }
                    }
                )
                found = True
                break

        if not found:
            raise RuntimeError(
                f"Unable to generate a reachable {task_kind} task after {max_retry_per_task} retries."
            )

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate tasks and append into data.json"
    )
    parser.add_argument(
        "--layout-key",
        default=DEFAULT_LAYOUT_KEY,
        type=str,
        help="Layout key name in data.json/data.py, e.g. layout_list_15/layout_list_30",
    )
    parser.add_argument(
        "--tasks-key",
        default=DEFAULT_TASKS_KEY,
        type=str,
        help="Task key name in data.json, generated tasks will append to this key",
    )
    parser.add_argument(
        "--task_count", default=50, type=int, help="Number of tasks to generate"
    )
    parser.add_argument(
        "--veh-name", type=str, default="veh1", help="Vehicle name in task dict"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    args = parser.parse_args()

    layout_list = load_layout_from_sources(args.layout_key)
    tasks = generate_tasks(
        layout_list, args.task_count, veh_name=args.veh_name, seed=args.seed
    )
    total_num = append_tasks_to_data_json(args.tasks_key, tasks)
    set_latest_keys(args.layout_key, args.tasks_key)

    print(
        f"Appended {len(tasks)} tasks to data.json key '{args.tasks_key}'. Current total: {total_num}"
    )
    print(json.dumps(tasks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
