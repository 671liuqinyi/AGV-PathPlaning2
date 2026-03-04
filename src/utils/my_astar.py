import heapq
import math

import numpy as np


class Cell:
    """节点类，增加 __lt__ 以支持堆排序直接比较"""

    def __init__(self, position=(0, 0), g=0, h=0, parent=None):
        self.position = position
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    # 重载比较运算符，当 f 相等时，g 较大的节点（更靠近目标的）优先
    def __lt__(self, other):
        if self.f == other.f:
            return self.g > other.g
        return self.f < other.f


class Gridworld:
    def __init__(self, world_map):
        self.wm = np.array(world_map)
        self.y_limit, self.x_limit = self.wm.shape

    def get_neighbours(self, cell):
        """获取四连通邻居"""
        neighbours = []
        # 上下左右
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        curr_x, curr_y = cell.position

        for dx, dy in directions:
            nx, ny = curr_x + dx, curr_y + dy
            # 边界检查 + 障碍物检查 (1.0 为通路)
            if 0 <= nx < self.x_limit and 0 <= ny < self.y_limit and self.wm[ny][nx] == 1.0:
                neighbours.append((nx, ny))
        return neighbours


class FindPathAstar:
    def __init__(self, world_map, start_pos, target_pos):
        self.grid = Gridworld(world_map)
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.path_list = []
        self.action_list = []
        self.scanned_count = 0  # 初始化计数器

    def _manhattan_dist(self, pos1, pos2):
        """曼哈顿距离启发式函数"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        # 欧式距离
        # return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        # 论文原代码使用距离
        # return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    def run(self):
        """主运行入口"""
        found = self.search()
        if found:
            self._generate_actions()
            return True, self.path_list, self.grid.wm, self.action_list
        return False, [], self.grid.wm, []

    def search(self):
        # 优先级队列存储 (f_score, Cell对象)
        open_heap = []
        # 记录已访问节点及其最小 g 值的映射，优化查询
        open_dict = {}
        closed_set = set()

        # 初始化起点
        start_h = self._manhattan_dist(self.start_pos, self.target_pos)
        start_node = Cell(self.start_pos, g=0, h=start_h)

        heapq.heappush(open_heap, start_node)
        open_dict[self.start_pos] = start_node.g

        while open_heap:
            # 每次从 open 表取出一个节点，计数加 1
            self.scanned_count += 1
            # 获取当前 f 值最小的节点
            current = heapq.heappop(open_heap)

            # 如果已经到达目标
            if current.position == self.target_pos:
                self._reconstruct_path(current)
                return True

            if current.position in closed_set:
                continue

            closed_set.add(current.position)

            for next_pos in self.grid.get_neighbours(current):
                if next_pos in closed_set:
                    continue

                new_g = current.g + 1

                # 如果当前路径更优，则更新该节点
                if next_pos not in open_dict or new_g < open_dict[next_pos]:
                    h = self._manhattan_dist(next_pos, self.target_pos)
                    neighbor_node = Cell(next_pos, g=new_g, h=h, parent=current)

                    open_dict[next_pos] = new_g
                    heapq.heappush(open_heap, neighbor_node)

        return False

    def _reconstruct_path(self, end_node):
        """从终点回溯生成路径"""
        curr = end_node
        while curr:
            self.path_list.append(curr.position)
            # 在地图上标记路径（可选）
            self.grid.wm[curr.position[1]][curr.position[0]] = -1
            curr = curr.parent
        self.path_list.reverse()  # 转为从起点到终点的顺序

    def _generate_actions(self):
        """生成动作指令序列"""
        # 定义移动映射
        move_map = {
            (1, 0): "RIGHT",
            (-1, 0): "LEFT",
            (0, 1): "DOWN",
            (0, -1): "UP"
        }
        for i in range(len(self.path_list) - 1):
            p1 = self.path_list[i]
            p2 = self.path_list[i + 1]
            vector = (p2[0] - p1[0], p2[1] - p1[1])
            self.action_list.append(move_map.get(vector, "UNKNOWN"))


# --- 测试用例 ---
if __name__ == "__main__":
    valid_path = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]

    start_point = (0, 3)
    target_point = (5, 5)


    finder = FindPathAstar(valid_path, start_point, target_point)
    success, path, res_map, actions = finder.run()

    if success:
        print(f"成功找到路径！步数: {len(path) - 1}, 扫描节点数: {finder.scanned_count}")
        print(f"动作序列: {actions}")
        print("最终地图分布 (-1位路径):")
        print(res_map)
    else:
        print("无法到达目标。")
