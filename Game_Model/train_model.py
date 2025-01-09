
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

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(device)

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file
config.read('config.ini')

# Access values from the configuration file
EX_MODEL = config.getboolean('Params', 'EX_MODEL')
BATCH_SIZE = config.getint('Params', 'BATCH_SIZE')
GAMMA = config.getfloat('Params', 'GAMMA')
EPS_START = config.getfloat('Params', 'EPS_START')
EPS_END = config.getfloat('Params', 'EPS_END')
EPS_DECAY = config.getfloat('Params', 'EPS_DECAY')
LR = config.getfloat('Params', 'LR')
TARGET_Q = config.getint('Params', 'TARGET_Q')

"""
    ReplayMemory this keeps track of preiovus moves for training and is used to train ght emodel
"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

if not EX_MODEL: # 2D CNN
    policy_net = DQN(11).to(device) # F(State, Action) -> Reward
    target_net = DQN(11).to(device) # Maximize Reward
else: # 3D CNN
    policy_net = DQNex(1).to(device) # F(State, Action) -> Reward
    target_net = DQNex(1).to(device) # Maximize Reward
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0).unsqueeze(1)
    reward_batch = torch.cat(batch.reward, dim=0)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # outputs 32x4 tensor
    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((BATCH_SIZE), device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # print(state_action_values, expected_state_action_values)

    # Compute MSE loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10) # normal clipping was 100
    optimizer.step()

    return loss.item()


num_episodes = 500 # epochs

model_loss = []

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    g = game.Game(multidimensional=True)
    num_episodes_w_no_movement = 0

    state = g.board.copy()
    # state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0) # nn
    if not EX_MODEL:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # cnn
    else:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # 3d cnn

    for t in count():
        prev = g.board.copy()

        action, steps_done = select_action(prev, policy_net, steps_done, EPS_START, EPS_END, EPS_DECAY, EX_MODEL)
        
        # print('move is', action,'type is', type(action), end = "\r")
        success, combined_values, combined_indexes = g.move(action)
        action = torch.tensor([action], dtype=torch.int64, device=device)
        observation = g.board.copy()
        
        terminated = g.check_win()
        truncated = g.check_loss()

        reward = get_reward(prev, action, observation, terminated, truncated, success, combined_values, combined_indexes)
        # print('reward is', reward, 'won: ', terminated, 'lost: ', truncated, end="\r")
        
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        done = terminated or truncated

        if truncated:
            next_state = None
        else:
            # next_state = torch.tensor(observation.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
            if not EX_MODEL:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) # cnn
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # 3d cnn

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
        model_loss.append(loss)

        # Soft update of the target network's weights
        if steps_done % TARGET_Q == 0:
            print('updating static network')
            policy_net_state_dict = policy_net.state_dict()
            target_net.load_state_dict(policy_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break

    print('done with epoch #{}, max: {}, loss: {}, steps played: {}'.format(i_episode, g.max(), loss, episode_durations[-1]), end='\r')

plt.plot(model_loss)
plt.show()

# saving the models
target_net_state_dict = target_net.state_dict()
policy_net_state_dict = policy_net.state_dict()

if not EX_MODEL: # 2D CNN
    torch.save(target_net_state_dict, 'Model/saved_models/2dtarget_net.pt')
    torch.save(policy_net_state_dict, 'Model/saved_models/2dpolicy_net.pt')
else: # 3D CNN
    torch.save(target_net_state_dict, 'Model/saved_models/2dtarget_net.pt')
    torch.save(policy_net_state_dict, 'Model/saved_models/2dpolicy_net.pt')







