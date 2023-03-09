import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_durations(duration: list, file_path: str):
    """
    plot total durations
    """
    plt.figure(1)
    duration_torch = torch.tensor(duration, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Step')
    plt.title("Number of steps in episodes")
    plt.plot(duration_torch.numpy())
    means = duration_torch.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy())
    # plt.savefig(file_path)
    plt.show()
    plt.close()

def plot_num_success_bar(num_success: int, time: int, file_path: str):
    """
    """
    plt.figure(1)
    plt.ylabel('Number')
    plt.title("Number of successful and unsuccessful episodes getting the frisbee")
    plt.bar(['Success', 'Failure'], [num_success, time - num_success])
    # plt.savefig(file_path)
    plt.show()
    plt.close()

def plot_reward(reward_numpy: np.ndarray, file_path: str):
    """
    """
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title("Reward of episode")
    x = np.arange(0, reward_numpy.shape[0])
    y = reward_numpy
    reward_torch = torch.tensor(reward_numpy, dtype=torch.float)
    means = reward_torch.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy(), color='r')
    plt.scatter(x, y, s=1)
    # plt.savefig(file_path)
    plt.show()
    plt.close()