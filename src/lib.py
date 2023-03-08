import numpy as np

def variables(epsilon: float):
    """
    return variables need
    return:
        ALL_ACTIONS, actions, ALL_POLICE, TERMINAL_AREA, NEGATIVE1_AREA
    """
    e = epsilon

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

    return ALL_ACTIONS, actions, ALL_POLICE

def areas(map_size: int):
    """
    return:
        TERMINAL_AREA, NEGATIVE1_AREA
    """
    if map_size == 4:
        TERMINAL_AREA = [(1,1), (1,3), (2,3), (3,0), (3,3)]
        NEGATIVE1_AREA = [(1,1), (1,3), (2,3), (3,0)]
    else:
        TERMINAL_AREA = [(0,2),(0,5),(1,3),(1,6),(1,9),(2,0),(2,4),(2,7),(3,0),(3,4),(4,1),(4,5),(5,2),(5,6),
                          (6,1),(6,3),(6,5),(7,3),(7,6),(7,7),(8,2),(8,4),(8,6),(8,8),(9,4),(9,9)]
        NEGATIVE1_AREA = [(0,2),(0,5),(1,3),(1,6),(1,9),(2,0),(2,4),(2,7),(3,0),(3,4),(4,1),(4,5),(5,2),(5,6),
                          (6,1),(6,3),(6,5),(7,3),(7,6),(7,7),(8,2),(8,4),(8,6),(8,8),(9,4)]

    return TERMINAL_AREA, NEGATIVE1_AREA


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

def random_action(current_policy: dict, e: float):
    """
    randomly choose an action based on epsilon greedy policy
    """
    p = np.random.random()
    max_action, _ = optimal_action(current_policy=current_policy)
    ALL_ACTIONS, _, _ = variables(epsilon=e)

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

def generate1episode(policies: np.ndarray, size: int, e: float, map_array: np.ndarray):
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
    _, actions, _ = variables(epsilon=e)
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
            reward = map_array[next_state[0], next_state[1]]
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
    4 * size * size three dimensional array.
    first: up, down, right, left
    second: x, y
    """
    table = np.zeros((4, size, size))
    return table

def find_position(action: str, all_actions: tuple) -> int:
    """
    find index in the ALL_ACTION tuple
    """
    res = -10
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
    ALL_ACTIONS, _, _ = variables(epsilon=epsilon)
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
    TERMINAL_AREA, NEGATIVE1_AREA = areas(map_size=size)

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
    TERMINAL_AREA, NEGATIVE1_AREA = areas(map_size=size)
    if x < 0 or x >= size or y < 0 or y >= size or state in NEGATIVE1_AREA:
        return True
    else:
        return False
    
def createReturnsList(size: int, epsilon: float) -> list:
    """
    create Returns list
    """
    ALL_ACTIONS, _, _ = variables(epsilon=epsilon)
    res = []
    for i in range(size):
        for j in range(size):
            for action in ALL_ACTIONS:
                current_dict = {str(i)+','+str(j)+','+action: []}
                res.append(current_dict)
    return res

def have_SAbefore(previous_episode: list, state: str, action: str) -> bool:
    """
    For First visit monte carlo prediction, check whether current pair (S,A) 
    appears in previous sequence.
    Return 
    --------
    if have the same (State, Action) before, return true
    else return false
    """
    for step in previous_episode:
        current_state = list(step.keys())[0]
        current_action = step[current_state]
        if current_state == state and current_action == action:
            return True
    return False

def new_entry_pi(max_action: str, epsilon: float):
    """

    """
    max_new_policy = 1 - epsilon + epsilon/4
    other_new_policy = epsilon / 4
    res = {}
    ALL_ACTIONS, _, _ = variables(epsilon=epsilon)
    for action in ALL_ACTIONS:
        if action == max_action:
            res.update({action: max_new_policy})
        else:
            res.update({action: other_new_policy})
    return res