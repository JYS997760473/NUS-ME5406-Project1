from src.lib import *
from src.evaluation import *
import time as ttime

def Qlearning(size: int, epsilon: float, gamma: float=0.9, time: int = 1000):
    """
    Qlearning to get optimal policy
    """
    ALL_ACTIONS, actions, ALL_POLICE = variables(epsilon=epsilon)
    # initialize a Q-table
    Qtable = createQtable(size=size)
    # Qtable[1,0,0]=1
    # Qtable[2,1,0]=1
    # Qtable[1,1,1]=1
    # Qtable[1,2,1]=1
    # Qtable[2,3,1]=1
    # Qtable[2,3,2]=1
    # Qtable[2,4,2]=1
    # Qtable[1,4,3]=1
    # Qtable[1,5,4]=1
    times = 0
    duration = []
    reward_numpy = np.full((time), -1)
    num_success = 0
    while times < time:
        # initialize state
        current_coordinate = (0, 0)
        k = 0
        # start_time = ttime.time()
        print(times)
        print(f"success:{num_success}")
        while not check_state_terminal(state=current_coordinate, size=size):
            # create current policy based current Qtable
            current_policy = get_policy_from_Qtable(Qtable=Qtable, epsilon=epsilon, size=size)
            # choose action from Q based epsilon-greedy policy
            current_action = random_action(current_policy=current_policy[int(current_coordinate[0]), 
                                                            int(current_coordinate[1])], e=epsilon)
            next_state = np.array([int(current_coordinate[0]), int(current_coordinate[1])]) + \
                                            actions[current_action]
            x = int(current_coordinate[0])
            y = int(current_coordinate[1])
            x_next = next_state[0]
            y_next = next_state[1]
            Q_current = Qtable[find_position(current_action, ALL_ACTIONS), x, y]
            # if k == 0:
            #     alpha = 1
            # else:
            #     alpha = 1/k
            alpha = 0.2
            # get reward
            # check if -1
            if check_reward_negative1(state=(next_state[0], next_state[1]), size=size):
                reward = -1
            # check if is the 1
            elif next_state[0] == size-1 and next_state[1] == size-1:
                reward = 1
                reward_numpy[times] = 1
                num_success += 1
            else:
                reward = 0
            # choose next greedy action from next state 
            if reward == -1 or reward == 1:
                Q_next = 0
            else:
                next_action, _ = optimal_action(current_policy=current_policy[next_state[0], 
                                                                              next_state[1]])
                Q_next = Qtable[find_position(next_action, ALL_ACTIONS), x_next, y_next]
            tmp = alpha * (reward + gamma * Q_next - Q_current)
            Qtable[find_position(current_action, ALL_ACTIONS), x, y] = Q_current + tmp
            k += 1
            # update current coordinate
            current_coordinate = (next_state[0], next_state[1])
        # print(f"time use:{ttime.time() - start_time}")
        times += 1
        duration.append(k+1)
    optimal_policy = get_policy_from_Qtable(Qtable=Qtable, epsilon=epsilon, size=size)
    return Qtable, optimal_policy, duration, reward_numpy, num_success