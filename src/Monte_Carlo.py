from src.lib import *
from src.evaluation import *

def monteCarlo(size: int, epsilon: float, map_array: np.ndarray, gamma: float, time: int):
    """
    first visit monte carlo control without exploring state
    """
    ALL_ACTIONS, actions, ALL_POLICE = variables(epsilon=epsilon)
    # initilaize an arbitrarily epsilon policy
    policies = create_random_policy(all_policy=ALL_POLICE, size=size)
    # print(f"first:{policies}")
    # initilaize a Q-table
    Qtable = createQtable(size=size)
    # returns stores G
    returns = createReturnsList(size=size, epsilon=epsilon)
    times = 0
    duration = []
    reward_numpy = np.full((time), -1)
    num_success = 0
    while times < time:
        print(times)
        episode, steps, valid = generate1episode(policies=policies, size=size, e=epsilon,
                                                 map_array=map_array)
        if episode[-1][list(episode[-1].keys())[0]][1] == 1:
            num_success += 1
            reward_numpy[times] = 1
        G = 0
        k = 0
        # backward iterate the episode
        for current_step in reversed(episode):
            # current coordinate
            current_coordinate = list(current_step.keys())[0]
            reward = current_step[current_coordinate][1]
            current_action = current_step[current_coordinate][0]
            G = gamma * G + reward
            # For first visit monte carlo, check if this (state, action) pair is the first one
            if not have_SAbefore(previous_episode=episode[:steps], state=current_coordinate, action=current_action):
                steps -= 1
                # if current (state, action) pair is the first one, append G to returns(state, action)
                x = int(current_coordinate[0])
                y = int(current_coordinate[2])
                index = (size * len(ALL_ACTIONS)) * x + len(ALL_ACTIONS) * y + find_position(current_action,
                                                                                              ALL_ACTIONS)
                # append G to Return list
                returns[index][current_coordinate+','+current_action].append(G)
                # set Q table
                Qtable[find_position(current_action, ALL_ACTIONS), x, y] = sum(returns[index][current_coordinate+
                                    ','+current_action]) / len(returns[index][current_coordinate+','+current_action])
                # choose the maximum action of the same state based Q table
                Q_max = -1000
                A = ""
                for i in range(4):
                    # iterate four actions
                    if Qtable[i, x, y] > Q_max:
                        Q_max = Qtable[i, x, y]
                        A = ALL_ACTIONS[i]
                # reset the current state's policy
                new_current_pi = new_entry_pi(max_action=A, epsilon=epsilon)
                policies[x, y] = new_current_pi
            k += 1
        duration.append(k)            
        times += 1
    return policies, Qtable, duration, reward_numpy, num_success