"""
Training entry for MAPPO on multi-AGV path planning.
"""

import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_JSON_PATH = os.path.join(PROJECT_ROOT, "data.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multiAGVscene.Layout import Layout
from multiAGVscene.Explorer import Explorer
from multiAGVscene.Scene import Scene
from algorithm.Manager.ExpertManager import Expert as Expert
from algorithm.MAPPO_structure.Controller import MAPPOAgentController as modelController

from data import layout_list_30


# 30x30 + 3AGV training speed/quality balance:
# train easy first, then increase difficulty.
TASK_NUM_LIMIT = 20
MAX_STEPS_PER_EPISODE = 3600
CURRICULUM_STAGES = [
    {"end_episode": 200, "task_num_limit": 20, "max_steps": 3600},
    {"end_episode": 500, "task_num_limit": 40, "max_steps": 5400},
    {"end_episode": 800, "task_num_limit": 60, "max_steps": 7200},
]


def main():
    # 1. Create layout
    ss_x_width, ss_y_width, ss_x_num, ss_y_num, ps_num = 8, 2, 3, 6, 3
    layout = Layout(
        storage_station_x_width=ss_x_width,
        storage_station_y_width=ss_y_width,
        storage_station_x_num=ss_x_num,
        storage_station_y_num=ss_y_num,
        picking_station_number=ps_num,
        layout_list=layout_list_30,
        task_list=None,
        task_num_limit=TASK_NUM_LIMIT,
    )

    # 2. Create AGVs
    explorer_num = 3
    explorer_group = []
    for i in range(explorer_num):
        veh_name = "veh" + str(i + 1)
        explorer_group.append(Explorer(layout, veh_name=veh_name, icon_name=veh_name))

    # 3. Create scene
    multi_agv_scene = Scene(layout, explorer_group)
    # Cap per-episode steps for faster iteration on large map.
    multi_agv_scene.max_training_steps = min(multi_agv_scene.max_training_steps, MAX_STEPS_PER_EPISODE)
    print(
        f"MAPPO train config: layout={layout.scene_x_width}x{layout.scene_y_width}, "
        f"agv={explorer_num}, task_num_limit={TASK_NUM_LIMIT}, max_steps={multi_agv_scene.max_training_steps}"
    )
    print(f"MAPPO curriculum stages: {CURRICULUM_STAGES}")

    # 4. Choose mode
    control_type = {0: "train_NN", 1: "use_NN", 2: "A_star", 3: "manual", 4: "Expert"}
    control_mode = 0

    print("Model is controlled by %s mode" % control_type[control_mode])

    if control_mode in [2, 3]:
        multi_agv_scene.run_game(control_pattern=control_type[control_mode])
        return

    if control_mode in [4]:
        expert = Expert(
            multi_agv_scene,
            ss_x_width,
            ss_y_width,
            ss_x_num,
            ss_y_num,
            ps_num,
            explorer_num,
        )
        expert.create_data_by_self(times=750)
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
        curriculum_stages=CURRICULUM_STAGES,
    )
    agent.model_run(run_mode=control_type[control_mode])


if __name__ == "__main__":
    main()
