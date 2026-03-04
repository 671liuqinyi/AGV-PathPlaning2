# src/utils/path_utils.py
import os
import re
def get_src_dir():
    """获取src根目录（当前运行路径）"""
    current_path = os.path.abspath(__file__)
    # 回退到src目录（utils是src的子目录）
    src_dir = os.path.abspath(os.path.join(current_path, "../"))
    return src_dir

def get_target_dir(sub_dir):
    """
    获取目标子目录（基于src）
    :param sub_dir: 相对src的子目录，如 "algorithm/DQN_structure/network_picture"
    :return: 目标目录绝对路径
    """
    src_dir = get_src_dir()
    target_dir = os.path.join(src_dir, sub_dir)
    # 确保目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir


def get_next_run_dir(base_train_path):
    """
    自动生成递增的run文件夹（run1, run2, run3...）
    :param base_train_path: 基础train目录路径
    :return: 新的run文件夹完整路径
    """
    # 确保基础train目录存在
    if not os.path.exists(base_train_path):
        os.makedirs(base_train_path)

    # 遍历现有文件夹，匹配run+数字的格式
    run_dirs = []
    for dir_name in os.listdir(base_train_path):
        dir_path = os.path.join(base_train_path, dir_name)
        if os.path.isdir(dir_path) and re.match(r'^run\d+$', dir_name):
            # 提取数字部分并转换为整数
            num = int(dir_name.replace('run', ''))
            run_dirs.append(num)

    # 计算下一个序号（无则从1开始）
    next_num = max(run_dirs) + 1 if run_dirs else 1
    next_run_dir = os.path.join(base_train_path, f'run{next_num}')

    # 创建新的run文件夹
    os.makedirs(next_run_dir, exist_ok=True)
    return next_run_dir