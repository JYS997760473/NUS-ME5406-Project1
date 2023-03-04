import numpy as np

# 4 * 4 map
map4 = np.array([[0, 0, 0, 0],
                 [0, -1, 0, -1],
                 [0, 0, 0, -1],
                 [-1, 0, 0, 1]])

# 10 * 10 map
map10 = np.array([[]])

# epsilon of epsilon-soft policy
e = 0.1

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

def createReturnsList(size: int) -> list:
    """
    create Returns list
    """
    res = []
    for i in range(size):
        for j in range(size):
            for action in ALL_ACTIONS:
                current_dict = {str(i)+','+str(j)+','+action: []}
                res.append(current_dict)
    return res

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

def new_entry_pi(max_action: str, epsilon: float):
    """

    """
    max_new_policy = 1 - epsilon + epsilon/4
    other_new_policy = epsilon / 4
    res = {}
    for action in ALL_ACTIONS:
        if action == max_action:
            res.update({action: max_new_policy})
        else:
            res.update({action: other_new_policy})
    return res

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



if __name__ == "__main__":
    policy, Qtable = monteCarlo(4, e, 0.9, 10000)
    print(f"policy:{policy}")
    print(f"Qtable:{Qtable}")
    # print(create_random_policy(ALL_POLICE, 4))