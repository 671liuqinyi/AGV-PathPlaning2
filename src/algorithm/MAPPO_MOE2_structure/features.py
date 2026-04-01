import numpy as np


ACTION_TO_OFFSET = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
EXPERT_FEATURE_DIM = 19


def _safe_cell(valid_path_matrix, x_pos, y_pos):
    map_h = len(valid_path_matrix)
    map_w = len(valid_path_matrix[0])
    if x_pos < 1 or y_pos < 1 or x_pos > map_w or y_pos > map_h:
        return 0.0
    return 1.0 if valid_path_matrix[y_pos - 1][x_pos - 1] != 0 else 0.0


def _manhattan(p1, p2):
    return abs(int(p1[0]) - int(p2[0])) + abs(int(p1[1]) - int(p2[1]))


def extract_expert_features(all_info, this_veh, valid_path_matrix, map_xdim, map_ydim):
    current_place = None
    target_place = None
    other_positions = []

    for idx in range(1, len(all_info)):
        veh_name, veh_current, veh_target, _ = all_info[idx]
        if veh_name == this_veh:
            current_place = list(veh_current)
            target_place = list(veh_target)
        else:
            other_positions.append(list(veh_current))

    if current_place is None or target_place is None:
        raise ValueError(f"extract_expert_features: vehicle '{this_veh}' not found in all_info")

    map_xdim = max(int(map_xdim), 1)
    map_ydim = max(int(map_ydim), 1)
    norm_dist = float(map_xdim + map_ydim)

    dx = float(target_place[0] - current_place[0]) / float(map_xdim)
    dy = float(target_place[1] - current_place[1]) / float(map_ydim)
    dist_norm = float(_manhattan(current_place, target_place)) / norm_dist

    free_flags = []
    occ_flags = []
    conflict_risk = []
    other_count = max(len(other_positions), 1)
    other_set = {(pos[0], pos[1]) for pos in other_positions}

    for action in range(4):
        offset = ACTION_TO_OFFSET[action]
        next_x = current_place[0] + offset[0]
        next_y = current_place[1] + offset[1]
        free_flags.append(_safe_cell(valid_path_matrix, next_x, next_y))

        occ_flag = 1.0 if (next_x, next_y) in other_set else 0.0
        occ_flags.append(occ_flag)

        if next_x < 1 or next_y < 1 or next_x > map_xdim or next_y > map_ydim:
            conflict_risk.append(1.0)
            continue

        risk_count = 0
        for other_pos in other_positions:
            if _manhattan([next_x, next_y], other_pos) <= 1:
                risk_count += 1
        conflict_risk.append(min(1.0, float(risk_count) / float(other_count)))

    if other_positions:
        nearest = min(other_positions, key=lambda p: _manhattan(current_place, p))
        near_dx = float(nearest[0] - current_place[0]) / float(map_xdim)
        near_dy = float(nearest[1] - current_place[1]) / float(map_ydim)
        near_dist = float(_manhattan(current_place, nearest)) / norm_dist
    else:
        near_dx = 0.0
        near_dy = 0.0
        near_dist = 1.0

    open_ratio = float(sum(free_flags)) / 4.0

    feature_vec = [
        dx,
        dy,
        dist_norm,
        free_flags[0],
        free_flags[1],
        free_flags[2],
        free_flags[3],
        occ_flags[0],
        occ_flags[1],
        occ_flags[2],
        occ_flags[3],
        near_dx,
        near_dy,
        near_dist,
        conflict_risk[0],
        conflict_risk[1],
        conflict_risk[2],
        conflict_risk[3],
        open_ratio,
    ]
    return np.array(feature_vec, dtype=np.float32)
