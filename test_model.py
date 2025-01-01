import torch
import numpy as np

import sys
from Model.cnnmodel import DQN

from reward_func import get_reward

from Game_Sim import game

def test_model(filename):
    model = DQN()
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    new_game = game.Game()
    while (not new_game.check_loss() and not new_game.check_win()):
        new_game.display()
        prev = new_game.board.copy()

        output = model(torch.tensor(new_game.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).detach().cpu()
        print(output)
        action = torch.argmax(output).item()
        print('action: ', action)
        successful, combined_values, combined_indexes = new_game.move(action)

        print('reward: ', get_reward(prev, action, new_game.board.copy(), new_game.check_win(), new_game.check_loss(), successful, combined_values, combined_indexes))
        next = input()
        if next == "close":
            break


if __name__ == "__main__":
    args = sys.argv
    test_model(args[1])

