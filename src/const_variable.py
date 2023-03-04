import numpy as np
from main import *

e = opt.epsilon

# e-soft policy
optimal = 1 - e + e/4
normal = e/4

ALL_ACTIONS = ("up", "down", "right", "left")

# actions in every state
actions = {"up": np.array([-1, 0]),
           "down": np.array([1, 0]),
           "right": np.array([0, 1]),
           "left": np.array([0, -1])}

# initialize an arbitrarily e-soft policy (4 * 4 dict array)
policy_down = {"up": normal,
                "down": optimal,
                "right": normal,
                "left": normal}
policy_right = {"up": normal,
                "down": normal,
                "right": optimal,
                "left": normal}
policy_up = {"up": optimal,
            "down": normal,
            "right": normal,
            "left": normal}
policy_left = {"up": normal,
                "down": normal,
                "right": normal,
                "left": optimal}
ALL_POLICE = (policy_up, policy_down, policy_right, policy_left)

if opt.map_size == 4:
    TERMINAL_AREA = [(1,1), (1,3), (2,3), (3,0), (3,3)]
    NEGATIVE1_AREA = [(1,1), (1,3), (2,3), (3,0)]
else:
    TERMINAL_AREA = [(1,3), (2,7), (3,4), (4,1), (5,2), (5,6), (6,5), (7,3), (8,8), (9, 0), (9,9)]
    NEGATIVE1_AREA = [(1,3), (2,7), (3,4), (4,1), (5,2), (5,6), (6,5), (7,3), (8,8), (9, 0)]