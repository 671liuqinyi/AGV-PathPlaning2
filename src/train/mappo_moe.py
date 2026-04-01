"""
Training entry for MAPPO+MoE on multi-AGV path planning.
"""

import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.MAPPO_MOE_structure.Controller import MAPPOMoEAgentController as modelController
from multiAGVscene.Explorer import Explorer
from multiAGVscene.Layout import Layout
from multiAGVscene.Scene import Scene

from data import layout_list_15, layout_list_30


# Choose training layout: "15" or "30"
LAYOUT_MODE = "30"

CONFIG_BY_LAYOUT = {
    "15": {
        "layout_list": layout_list_15,
        "task_num_limit": 10,
        "max_steps_per_episode": 1800,
        "curriculum_stages": [
            {"end_episode": 200, "task_num_limit": 10, "max_steps": 1800},
            {"end_episode": 500, "task_num_limit": 16, "max_steps": 2400},
            {"end_episode": 800, "task_num_limit": 24, "max_steps": 3000},
        ],
    },
    "30": {
        "layout_list": layout_list_30,
        "task_num_limit": 20,
        "max_steps_per_episode": 3600,
        "curriculum_stages": [
            {"end_episode": 200, "task_num_limit": 20, "max_steps": 3600},
            {"end_episode": 500, "task_num_limit": 40, "max_steps": 5400},
            {"end_episode": 800, "task_num_limit": 60, "max_steps": 7200},
        ],
    },
}


def main():
    if LAYOUT_MODE not in CONFIG_BY_LAYOUT:
        raise ValueError(f"Unsupported LAYOUT_MODE: {LAYOUT_MODE}, expected one of {list(CONFIG_BY_LAYOUT.keys())}")
    cfg = CONFIG_BY_LAYOUT[LAYOUT_MODE]

    task_num_limit = cfg["task_num_limit"]
    max_steps_per_episode = cfg["max_steps_per_episode"]
    curriculum_stages = cfg["curriculum_stages"]

    ss_x_width, ss_y_width, ss_x_num, ss_y_num, ps_num = 8, 2, 3, 6, 3
    layout = Layout(
        storage_station_x_width=ss_x_width,
        storage_station_y_width=ss_y_width,
        storage_station_x_num=ss_x_num,
        storage_station_y_num=ss_y_num,
        picking_station_number=ps_num,
        layout_list=cfg["layout_list"],
        task_list=None,
        task_num_limit=task_num_limit,
    )

    explorer_num = 2
    explorer_group = []
    for i in range(explorer_num):
        veh_name = "veh" + str(i + 1)
        explorer_group.append(Explorer(layout, veh_name=veh_name, icon_name=veh_name))

    multi_agv_scene = Scene(layout, explorer_group)
    multi_agv_scene.max_training_steps = min(multi_agv_scene.max_training_steps, max_steps_per_episode)
    print(
        f"MAPPO+MoE train config: layout_mode={LAYOUT_MODE}, layout={layout.scene_x_width}x{layout.scene_y_width}, "
        f"agv={explorer_num}, task_num_limit={task_num_limit}, max_steps={multi_agv_scene.max_training_steps}"
    )
    print(f"MAPPO+MoE curriculum stages: {curriculum_stages}")

    control_type = {0: "train_NN", 1: "use_NN", 2: "A_star", 3: "manual"}
    control_mode = 0
    print("Model is controlled by %s mode" % control_type[control_mode])

    if control_mode in [2, 3]:
        multi_agv_scene.run_game(control_pattern=control_type[control_mode])
        return

    map_xdim = layout.scene_x_width
    map_ydim = layout.scene_y_width
    max_task = len(layout.storage_station_list)
    agent = modelController(
        multi_agv_scene,
        map_xdim=map_xdim,
        map_ydim=map_ydim,
        max_task=max_task,
        control_mode=control_type[control_mode],
        state_number=3,
        curriculum_stages=curriculum_stages,
    )
    agent.model_run(run_mode=control_type[control_mode])


if __name__ == "__main__":
    main()
