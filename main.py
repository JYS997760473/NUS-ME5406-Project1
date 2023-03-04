import argparse
from src.map import map
from src.Monte_Carlo import monteCarlo
from src.Qlearning import Qlearning
from src.SARSA import SARSA


def main(opt):
    """
    this is the main process of this project.
    """
    task = opt.task
    map_size = opt.map_size
    epsilon = opt.epsilon
    gamma = opt.gamma
    time = opt.time
    threshold = opt.threshold
    # create reward map
    map_array = map(map_size=map_size)
    if task == "Monte_Carlo":
        policy, Qtable = monteCarlo(size=map_size, epsilon=epsilon, map_array=map_array,
                                     gamma=gamma, time=time)
    elif task == "SARSA":
        Qtable, policy, gap, time = SARSA(size=map_size, threshold=threshold, epsilon=epsilon,
                                            gamma=gamma, time=time)
    else:
        # Q-learning
        Qtable, policy, gap, time = Qlearning(size=map_size, threshold=threshold, epsilon=epsilon,
                                              gamma=gamma, time=time)
    print(Qtable)
    print(policy)
    print(time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="", help="Monte_Carlo, SARSA or Q-learning")
    parser.add_argument("--map_size", type=int, default=4, help="the size of the map, 4 or 10")
    parser.add_argument("--epsilon", type=float, help="epsilon for using epsilon-soft policy")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--time", type=int, help="the number of iterations")
    parser.add_argument("--threshold", type=float, default=0.01, help="threshold in SARSA \
                        and Q-learning")
    opt = parser.parse_args()
    assert opt.map_size == 4 or opt.map_size == 10, "the size must be 4 or 10"
    main(opt=opt)