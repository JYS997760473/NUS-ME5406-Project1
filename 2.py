import numpy as np
import copy

# 4 * 4 map
map4 = np.array([[0, 0, 0, 0],
                 [0, -1, 0, -1],
                 [0, 0, 0, -1],
                 [-1, 0, 0, 1]])

ALL_ACTIONS = ("up", "down", "right", "left")

# epsilon of epsilon-soft policy
e = 0.1

# e-soft policy
optimal = 1 - e + e/4
normal = e/4

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

TERMINAL_AREA = [(1,1), (1,3), (2,3), (3,0), (3,3)]

NEGATIVE1_AREA = [(1,1), (1,3), (2,3), (3,0)]

def create_random_policy(all_policy: tuple, size: int) -> np.ndarray:
    """
    randomly create a new policy
    """
    res = np.full((size, size), {}, dtype=dict)
    for i in range(size):
        for j in range(size):
            res[i, j] = np.random.choice(all_policy)
    return res

def optimal_action(current_policy: dict):
    """
    find the optimal action of current state
    """
    max_value = -1.0
    decision_action = ""
    for action in current_policy.keys():
        if current_policy[action] > max_value:
            decision_action = action
            max_value = current_policy[action]
    return decision_action, max_value

def random_action(current_policy: dict, e: float = e):
    """
    randomly choose an action based on epsilon greedy policy
    """
    p = np.random.random()
    max_action, _ = optimal_action(current_policy=current_policy)

    if p < 1 - e + e/4:
        return max_action
    else:
        tmp = list(ALL_ACTIONS)
        tmp.remove(max_action)
        return np.random.choice(tmp)
    
def check_state_valid(state: tuple, size: int) -> bool:
    """
    check if the state is in the map
    """
    x = state[0]
    y = state[1]
    if 0 <= x < size and 0 <= y < size:
        return True
    else:
        return False

def generate1episode(policies: np.ndarray, size: int, e: float=e):
    """
    generate an episode with T steps following the policies
    Return
    ---------
    res: one episode list with several dict:{state: [action, reward]}
    step: number of steps
    valid: judge whether the final state is in the map
    """
    res = []
    step = 0
    current_state = np.array([0, 0])
    valid = True
    while True:
        # follow current policy to go to next state
        current_policy = policies[current_state[0], current_state[1]]
        current_action = random_action(current_policy=current_policy, e=e)
        next_state = current_state + actions[current_action]

        # check next state whether should break
        if check_state_valid(state=next_state, size=size) == False:
            # if next state is not in the map
            valid = False
            reward = -1
            current_dict = {str(current_state[0])+','+str(current_state[1]):[current_action, reward]}
            res.append(current_dict)
            # break
        else:
            # next state is in the map
            # put current state and next state's reward into res list
            reward = map4[next_state[0], next_state[1]]
            current_dict = {str(current_state[0])+','+str(current_state[1]):[current_action, reward]}
            res.append(current_dict)
        # if next state go to hole or terminal, break
        if reward == -1 or reward == 1:
            break
        else:
            current_state = next_state
        step += 1


    return res, step, valid

def createQtable(size: int):
    """
    Arbitrarily create a Q-table.
    Return 
    ---------
    4 * 4 * 4 three dimensional array.
    first: up, down, right, left
    second: x, y
    """
    table = np.zeros((4, size, size))
    return table

def find_position(action: str, all_actions: tuple) -> int:
    """
    find index in the ALL_ACTION tuple
    """
    res = -1
    for index, a in enumerate(all_actions):
        if action == a:
            res = index
            break
    return res

def create1entry_policy_from_Qtable(Qtable: np.ndarray, epsilon: float, size: int, x: int, y: int):
    """
    create one entry's policy
    """
    res = {}
    # find the max action from Q table
    Q_max = -1000
    A = ""
    for i in range(4):
        # iterate four actions
        if Qtable[i, x, y] > Q_max:
            Q_max = Qtable[i, x, y]
            A = ALL_ACTIONS[i]
    max_policy = 1 - epsilon + epsilon/4
    other_policy = epsilon / 4
    for action in ALL_ACTIONS:
        if action == A:
            res.update({action: max_policy})
        else:
            res.update({action: other_policy})
    return res

def get_policy_from_Qtable(Qtable: np.ndarray, epsilon: float, size: int):
    """
    For SARSA and Q-learning, create the final epsilon-policy based Qtable.
    """
    res = np.full((size, size), {}, dtype=dict)
    for x in range(size):
        for y in range(size):
            res[x, y] = create1entry_policy_from_Qtable(Qtable=Qtable, epsilon=epsilon, size=size, x=x, y=y)
    return res

def check_state_terminal(state: tuple, size: int) -> bool:
    """
    check if the state is terminal
    """
    x = state[0]
    y = state[1]
    if 0 <= x < size and 0 <= y < size and state not in TERMINAL_AREA:
        return False
    else:
        return True

def check_reward_negative1(state: tuple, size: int) -> bool:
    """
    check if the reward is -1
    """
    x = state[0]
    y = state[1]
    if x < 0 or x >= size or y < 0 or y >= size or state in NEGATIVE1_AREA:
        return True
    else:
        return False

def SARSA(size: int, threshold: float, epsilon: float=e, gamma: float=0.9, time: int = 1000):
    """
    SARSA to get optimal policy
    """
    # initialize a Q-table
    Qtable = createQtable(size=size)
    times = 0
    gap = 100.000
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
        aftQtable = copy.deepcopy(Qtable)
        gap = abs((aftQtable - preQtable).sum())
    optimal_policy = get_policy_from_Qtable(Qtable=Qtable, epsilon=epsilon, size=size)
    return Qtable, optimal_policy, gap, times

def Qlearning(size: int, threshold: float, epsilon: float=e, gamma: float=0.9, time: int = 1000):
    """
    Qlearning to get optimal policy
    """
    # initialize a Q-table
    Qtable = createQtable(size=size)
    times = 0
    gap = 100.000
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
            # choose next greedy action from next state 
            if reward == -1 or reward == 1:
                Q_next = 0
            else:
                next_action, _ = optimal_action(current_policy=current_policy[next_state[0], next_state[1]])
                Q_next = Qtable[find_position(next_action, ALL_ACTIONS), x_next, y_next]
            tmp = alpha * (reward + gamma * Q_next - Q_current)
            Qtable[find_position(current_action, ALL_ACTIONS), x, y] = Q_current + tmp
            k += 1
            # update current coordinate
            current_coordinate = (next_state[0], next_state[1])
        times += 1
        aftQtable = copy.deepcopy(Qtable)
        gap = abs((aftQtable - preQtable).sum())
    optimal_policy = get_policy_from_Qtable(Qtable=Qtable, epsilon=epsilon, size=size)
    return Qtable, optimal_policy, gap, times


if __name__ == "__main__":
    # Qtable, policy, gap, time = SARSA(4, 0.001, 0.1, 0.9, 10000)
    Qtable, policy, gap, time = Qlearning(4, 0.001, 0.1, 0.9, 10000)
    print(Qtable)
    print(policy)
    print(gap)
    print(time)