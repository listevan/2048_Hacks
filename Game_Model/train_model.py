import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple, deque
import random
import math
import numpy as np
from itertools import count
import configparser

from Model.cnnmodel import DQN
from Model.cnnmodel_ex import DQN as DQNex

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from reward_func import get_reward
from select_action import select_action
from Game_Sim import game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file
config.read("Game_Model/config.ini")

# Access values from the configuration file
EX_MODEL = bool(input("Train 2D CNN Model (0) or 3D CNN Model (1): "))
BATCH_SIZE = config.getint("PARAMS", "BATCH_SIZE")
GAMMA = config.getfloat("PARAMS", "GAMMA")
EPS_START = config.getfloat("PARAMS", "EPS_START")
EPS_END = config.getfloat("PARAMS", "EPS_END")
EPS_DECAY = config.getint("PARAMS", "EPS_DECAY")
LR = config.getfloat("PARAMS", "LR")
TARGET_Q = config.getint("PARAMS", "TARGET_Q")
num_episodes = config.getint("PARAMS", "num_episodes")  # epochs

"""
    ReplayMemory this keeps track of preiovus moves for training and is used to train ght emodel
"""
Transition = namedtuple("Transition", ("state", "action", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if not EX_MODEL:  # 2D CNN
    policy_net = DQN(11).to(device)  # F(State, Action) -> Reward
    target_net = DQN(11).to(device)  # Maximize Reward
else:  # 3D CNN
    policy_net = DQNex(1).to(device)  # F(State, Action) -> Reward
    target_net = DQNex(1).to(device)  # Maximize Reward
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = np.concatenate(batch.state, axis=0)
    reward_batch = torch.zeros((BATCH_SIZE, 4), device = device)

    next_state_values = torch.zeros((BATCH_SIZE, 4), device=device)

    for i in range(4):
        with torch.no_grad():
            next_states = torch.zeros((BATCH_SIZE, 1, 11, 4, 4))
            for j in range(BATCH_SIZE):
                temp_g = game.Game(multidimensional=True)
                temp_g.from_board(state_batch[j].squeeze(), multidimensional=True)
                successful, c_v, c_i = temp_g.move(i)
                next_states[i, 0, :, :, :] = torch.tensor(g.board)

                reward_batch[j, i] = get_reward(
                    state_batch[j].squeeze(),
                    i,
                    temp_g.board,
                    temp_g.check_win(),
                    temp_g.check_loss(),
                    successful,
                    c_v,
                    c_i,
                )

            next_states = torch.tensor(next_states).to(device)
            next_state_values[:, i] = target_net(next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    state_action_values = policy_net(
        torch.tensor(state_batch, dtype=torch.float32, device=device)
    )

    # Compute MSE loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(
        policy_net.parameters(), 1
    )  # normal clipping was 100
    optimizer.step()

    return loss.item()

model_loss = []
model_scores = []

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots((2))
line, = ax[0].plot([], [], 'b-', label='Loss')
ax[0].set_xlim(0, 10)  # Initial x-axis range
ax[0].set_ylim(0, 2)   # Initial y-axis range (adjust as needed)
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')
ax[0].legend()

line2, = ax[1].plot([], [], 'b-', label='Max Score')
ax[1].set_xlim(0, 10)
ax[1].set_ylim(0, 2)
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Max Score')
ax[1].legend()
plt.grid()

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    g = game.Game(multidimensional=True)

    state = g.board
    # state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0) # nn
    if not EX_MODEL:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # cnn
    else:
        state = (
            torch.tensor(state, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # 3d cnn

    for t in count():
        prev = g.board

        action, steps_done = select_action(
            prev, policy_net, steps_done, EPS_START, EPS_END, EPS_DECAY, EX_MODEL
        )

        # print('move is', action,'type is', type(action), end = "\r")
        success, combined_values, combined_indexes = g.move(action)
        action = torch.tensor([action], dtype=torch.int64, device=device)
        observation = g.board

        terminated = g.check_win()
        truncated = g.check_loss()

        reward = get_reward(
            prev,
            action,
            observation,
            terminated,
            truncated,
            success,
            combined_values,
            combined_indexes,
        )
        # print('reward is', reward, 'won: ', terminated, 'lost: ', truncated, end="\r")

        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        done = terminated or truncated

        if truncated:
            next_state = None
        else:
            # next_state = torch.tensor(observation.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
            if not EX_MODEL:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(
                    0
                )  # cnn
            else:
                next_state = (
                    torch.tensor(observation, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                # 3d cnn

        # Store the transition in memory
        memory.push(
            np.expand_dims(np.expand_dims(prev, axis=0), axis=0), action, reward
        )

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        m_loss = optimize_model()
        if m_loss is not None:
            model_loss.append(m_loss)
        
            line.set_ydata(model_loss)
            line.set_xdata(range(len(model_loss)))
            ax[0].set_xlim(0, len(model_loss))
            ax[0].set_ylim(0, max(model_loss) + 0.1)
            plt.draw()
            plt.pause(.1)

        # Soft update of the target network's weights
        if steps_done % TARGET_Q == 0:
            print("updating static network")
            policy_net_state_dict = policy_net.state_dict()
            target_net.load_state_dict(policy_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            model_scores.append(g.max())
            line2.set_ydata(model_scores)
            line2.set_xdata(range(len(model_scores)))
            ax[1].set_xlim(0, len(model_scores))
            ax[1].set_ylim(0, max(model_scores) + 1)
            plt.draw()
            break
    print(
        "[done with epoch #{}, max: {}, loss: {}, steps played: {}]".format(
            i_episode, g.max(), m_loss, episode_durations[-1]
        ),
        end="\r",
    ) 

plt.plot(model_loss)
plt.show()

# saving the models
target_net_state_dict = target_net.state_dict()
policy_net_state_dict = policy_net.state_dict()

if not EX_MODEL:  # 2D CNN
    torch.save(target_net_state_dict, "Model/saved_models/2dtarget_net.pt")
    torch.save(policy_net_state_dict, "Model/saved_models/2dpolicy_net.pt")
else:  # 3D CNN
    torch.save(target_net_state_dict, "Game_Model/Model/saved_models/2dtarget_net.pt")
    torch.save(policy_net_state_dict, "Game_Model/Model/saved_models/2dpolicy_net.pt")
