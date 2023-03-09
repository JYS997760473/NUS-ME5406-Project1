# NUS-ME5406-Project1

NUS ME5406 Deep Learning for Robotics Project 1

## Usage

### Step up python environment

Before run this project, please make sure that you have installed these packages in your environment:

- torch (only used to help plot figures of results)
- numpy
- matplotlib

### Run the code

After git clone or download this project, please run `main.py` file.

There are three reinforcement learning algorithms (Monte Carlo Control, SARSA and Q-learning), two size of maps ($4 \times 4$ and $10\times 10$), $\epsilon$, epsilon, $\gamma$, gamma and number of episodes you should set first. 

For example:

#### For SARSA with $4\times 4$ map, epsilon is 0.1, gamma is 0.9 and number of episodes is 10000:

```bash
python main.py --task SARSA --map_size 4 --epsilon 0.1 --gamma 0.9 --time 10000
```

#### For Monte Carlo with $4\times 4$ map, epsilon is 0.1, gamma is 0.9 and number of episodes is 10000:

```bash
python main.py --task Monte_Carlo --map_size 4 --epsilon 0.1 --gamma 0.9 --time 10000
```

#### For Q-learning with $4\times 4$ map, epsilon is 0.1, gamma is 0.9 and number of episodes is 10000:

```bash
python main.py --task Q-learning --map_size 4 --epsilon 0.1 --gamma 0.9 --time 10000
```

The options for `task` are `Monte_Carlo`, `SARSA` and `Q-learning`, for `map_size` is `4` and `10`. 

`epsilon`, `gamma` and `time` don't have fixed options, but if you want to try other parameters, please make sure that the parameters you choose are reasonable. 

### Result

After running the code, it will generate three figures showing the evaluation results of average step, numbers of successful and failed episodes bar and average reward.

For example: 

Run the code:

```bash
python main.py --task SARSA --map_size 4 --epsilon 0.1 --gamma 0.9 --time 10000
```

It will generate three evalution figures:

Average step:

<img src="/Users/jiayansong/Desktop/nus/ME5406/NUS-ME5406-Project1/figures/step.png" alt="isolated" width="400"/>

Number bar:

<img src="/Users/jiayansong/Desktop/nus/ME5406/NUS-ME5406-Project1/figures/bar.png" alt="isolated" width="400"/>

Average reward:

<img src="/Users/jiayansong/Desktop/nus/ME5406/NUS-ME5406-Project1/figures/reward.png" alt="isolated" width="400"/>