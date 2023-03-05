import numpy as np
import copy
from src.lib import *

def SARSA(size: int, threshold: float, epsilon: float, gamma: float=0.9, time: int = 1000):
    """
    SARSA to get optimal policy
    """
    ALL_ACTIONS, actions, ALL_POLICE = variables(epsilon=epsilon)
    # initialize a Q-table
    Qtable = createQtable(size=size)
    times = 0
    gap = 100.000
    duration = []
    while times < time or gap > threshold:
        # initialize state
        current_coordinate = (0, 0)
        k = 0
        preQtable = copy.deepcopy(Qtable)
        while not check_state_terminal(state=current_coordinate, size=size):
            # create current policy based current Qtable
            current_policy = get_policy_from_Qtable(Qtable=Qtable, epsilon=epsilon, size=size)
            # choose action from Q based epsilon-greedy policy
            current_action = random_action(current_policy=current_policy[int(current_coordinate[0]), int(current_coordinate[1])], e=epsilon)
            next_state = np.array([int(current_coordinate[0]), int(current_coordinate[1])]) + actions[current_action]
            x = int(current_coordinate[0])
            y = int(current_coordinate[1])
            x_next = next_state[0]
            y_next = next_state[1]
            Q_current = Qtable[find_position(current_action, ALL_ACTIONS), x, y]
            if k == 0:
                alpha = 1
            else:
                alpha = 1/k
            # get reward
            # check if -1
            if check_reward_negative1(state=(next_state[0], next_state[1]), size=size):
                reward = -1
            # check if is the 1
            elif next_state[0] == size-1 and next_state[1] == size-1:
                reward = 1
            else:
                reward = 0
            # choose next action from next state based current policy
            if reward == -1 or reward == 1:
                Q_next = 0
            else:
                next_action = random_action(current_policy=current_policy[next_state[0], next_state[1]], e=epsilon)  
                Q_next = Qtable[find_position(next_action, ALL_ACTIONS), x_next, y_next]
            tmp = alpha * (reward + gamma * Q_next - Q_current)
            Qtable[find_position(current_action, ALL_ACTIONS), x, y] = Q_current + tmp
            k += 1
            # update current coordinate
            current_coordinate = (next_state[0], next_state[1])
        times += 1
        duration.append(k+1)
        aftQtable = copy.deepcopy(Qtable)
        gap = abs((aftQtable - preQtable).sum())
    optimal_policy = get_policy_from_Qtable(Qtable=Qtable, epsilon=epsilon, size=size)
    plot_durations(duration=duration)
    return Qtable, optimal_policy, gap, times

