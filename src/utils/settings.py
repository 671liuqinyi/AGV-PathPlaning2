"""all specific settings"""


class SuperParas:
    # 启用各种起停、工作时间
    # activate all working times
    # Open_Work_Statue = True

    # explorer是否始终是空载或满载（能否从货架通行）
    # whether explorer is always in the state of be occupied or empty
    Explorer_Always_Loaded = False
    Explorer_Always_Empty = False

    # 屏幕刷新频率 FPS
    FPS = 300

    # sparse reward
    Sparse_Reward = False

    # Debug prints in training loop (collision/boundary logs). Keep False for speed.
    Debug_Train_Log = False

    # Lightweight dense reward (no A*): encourage moving toward target each step.
    Distance_Shaping_Reward = True
    Distance_Shaping_Positive = 0.02
    Distance_Shaping_Negative = -0.02
    Distance_Shaping_Stay = -0.003


