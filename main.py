import argparse
from src.map import map
from src.Monte_Carlo import monteCarlo
from src.Qlearning import Qlearning
from src.SARSA import SARSA
from src.evaluation import *
import os

def main(opt):
    """
    this is the main process of this project.
    """
    task = opt.task
    map_size = opt.map_size
    epsilon = opt.epsilon
    gamma = opt.gamma
    time = opt.time
    # create reward map
    map_array = map(map_size=map_size)
    if task == "Monte_Carlo":
        policy, Qtable, duration, reward_numpy, num_success = monteCarlo(size=map_size, epsilon=epsilon, 
                                                                         map_array=map_array, gamma=gamma, time=time)
    elif task == "SARSA":
        Qtable, policy, duration, reward_numpy, num_success = SARSA(size=map_size, 
                                                                        epsilon=epsilon, gamma=gamma, time=time)
    else:
        # Q-learning
        Qtable, policy, duration, reward_numpy, num_success = Qlearning(size=map_size,
                                                                        epsilon=epsilon, gamma=gamma, time=time)
        

    # target root
    target_root_directory = "/Users/jiayansong/Desktop/nus/ME5406/figures"
    target_directory = os.path.join(target_root_directory, task)

    # plot steps of episodes figure
    file_name = "map"+str(map_size)+'_e'+str(epsilon)+'_gamma'+str(gamma)+'_t'+str(time)+'.png'
    file_path = os.path.join(target_directory, file_name)
    plot_durations(duration=duration, file_path=file_path)

    # plot number of success bar
    bar_name = "map"+str(map_size)+'_e'+str(epsilon)+'_gamma'+str(gamma)+'_t'+str(time)+'_bar.png'
    bar_path = os.path.join(target_directory, bar_name)
    plot_num_success_bar(num_success=num_success, time=time, file_path=bar_path)

    # plot reward
    reward_name = "map"+str(map_size)+'_e'+str(epsilon)+'_gamma'+str(gamma)+'_t'+str(time)+'_reward.png'
    reward_path = os.path.join(target_directory, reward_name)
    plot_reward(reward_numpy=reward_numpy, file_path=reward_path)

    # record log file
    log_name = "map"+str(map_size)+'_e'+str(epsilon)+'_gamma'+str(gamma)+'_t'+str(time)+'_log.txt'
    log_path = os.path.join(target_directory, log_name)
    file = open(log_path, 'w+')
    file.write("Q table\n\n")
    file.write(str(Qtable)+'\n\n')
    file.write("optimal policy\n\n")
    file.write(str(policy))
    file.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="", help="Monte_Carlo, SARSA or Q-learning")
    parser.add_argument("--map_size", type=int, default=4, help="the size of the map, 4 or 10")
    parser.add_argument("--epsilon", type=float, help="epsilon for using epsilon-soft policy")
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--time", type=int, help="the number of iterations")
    opt = parser.parse_args()
    assert opt.map_size == 4 or opt.map_size == 10, "the size must be 4 or 10"
    main(opt=opt)