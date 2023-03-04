from src.const_variable import *
from src.lib import *

def monteCarlo(size: int, epsilon: float=e, gamma: float=0.9, time: int = 1000):
    """
    first visit monte carlo control without exploring state
    """
    # initilaize an arbitrarily epsilon policy
    policies = create_random_policy(all_policy=ALL_POLICE, size=size)
    print(f"first:{policies}")
    # initilaize a Q-table
    Qtable = createQtable(size=size)
    # returns stores G
    returns = createReturnsList(size=size)
    times = 0
    while times < time:
        # generate an valid episode with T steps folloing policy
        # while True:
        #     episode, steps, valid = generate1episode(policies=current_policies, size=size, e=epsilon)
        #     if valid:
        #         break
        episode, steps, valid = generate1episode(policies=policies, size=size, e=epsilon)
        # if episode[-1][list(episode[-1].keys())[0]][1] == 1:
            # print(f"got frisbee, time:{times}")
        G = 0
        # print(f"episode:{episode}")
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
                index = (size * len(ALL_ACTIONS)) * x + len(ALL_ACTIONS) * y + find_position(current_action, ALL_ACTIONS)
                # append G to Return list
                # print(f"RETURNS:{returns[index]}, index: {index}")
                returns[index][current_coordinate+','+current_action].append(G)
                # set Q table
                Qtable[find_position(current_action, ALL_ACTIONS), x, y] = sum(returns[index][current_coordinate+','+current_action]) / len(returns[index][current_coordinate+','+current_action])
                # choose the maximum action of the same state based Q table
                Q_max = -1000
                # print(f"Qtable:{Qtable}")
                A = ""
                for i in range(4):
                    # iterate four actions
                    if Qtable[i, x, y] > Q_max:
                        Q_max = Qtable[i, x, y]
                        A = ALL_ACTIONS[i]
                # reset the current state's policy
                new_current_pi = new_entry_pi(max_action=A, epsilon=epsilon)
                policies[x, y] = new_current_pi
                # print(f"x,y:{x},{y},action:{A}")
                # print("change")
                # print(policies)
                    
        times += 1
        # print(Qtable)
    return policies, Qtable
