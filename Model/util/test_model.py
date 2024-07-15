import torch
import numpy as np

import sys
sys.path.append("..") 
from model import DQN

sys.path.append("..")
from Game_Sim import game

def test_model(filename):
    model = DQN()
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    new_game = game.Game()
    while (not new_game.check_loss() and not new_game.check_win()):
        new_game.display()
        action = model(torch.tensor(new_game.board.flatten(), dtype=torch.float32))
        
        new_game.move(action)
        next = input()
        if next == "close":
            break

