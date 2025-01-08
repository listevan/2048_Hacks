import torch
import numpy as np
from itertools import count
import sys
from Model.cnnmodel import DQN
from Model.cnnmodel_ex import DQN as DQNex

from reward_func import get_reward

from Game_Sim.game import Game

def test_model(filename):
    model = DQNex(1)
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    new_game = Game(multidimensional=True)
    while (not new_game.check_loss() and not new_game.check_win()):
        new_game.display()
        backup = new_game.board.copy()
        prev = np.expand_dims(np.expand_dims(backup, 0), 0)

        action = select_action(prev, model).squeeze()
        print('action: ', action)
        successful, combined_values, combined_indexes = new_game.move(action)

        print('reward: ', get_reward(backup, action, new_game.board.copy(), new_game.check_win(), new_game.check_loss(), successful, combined_values, combined_indexes))
        next = input()
        if next == "stop":
            break

def select_action(state, model): # different from the one in the file, this is for testing and does an entire batch at a time
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = state.shape[0]
    model.to(device)
    
    input = torch.tensor(state, dtype=torch.float32, device=device)

    sorted_actions = torch.argsort(model(input).detach().cpu(), dim=-1) # normally is selected via policy_net
    
    final_actions = np.ones((batch_size)) * -1
    for g in range(batch_size):
        output = sorted_actions[g, :]
        for action in output:
            game = Game()
            game.from_board(state[g, :, :, :].squeeze(), multidimensional=True)
            success, _, _ = game.move(action)
            
            if not success:
                continue
            else:
                final_actions[g] = action
                break
        if final_actions[g] == -1:
            final_actions[g] = 0
    
    return final_actions

def max_value(game_board):
    game_board = game_board.copy().squeeze()
    # shape is 11 x 4 x 4
    for i in range(game_board.shape[0]-1, -1, -1):
        if (game_board[i, :, :] == 1).any():
            return 2**(i+1)
        
    return 0

def compare_model(base_cnn_filename, ex_cnn_filename, num_games=128, batch_size=128):
    if num_games % batch_size:
        raise Exception("invalid batch_size or num_games")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    nonlostgame = True # is there a game that is still in play

    model1 = DQN(nc=11) # 2d
    model1.load_state_dict(torch.load(base_cnn_filename, map_location=device))
    high_scores = np.zeros((num_games))
    finished_games = set()

    model2 = DQNex(nc=1) # 3d
    model2.load_state_dict(torch.load(ex_cnn_filename, map_location=device))
    high_scores2 = np.zeros((num_games))
    finished_games2 = set()

    for i in range(num_games // batch_size):
        states = np.zeros((batch_size, 11, 4, 4))

        for j in range(batch_size):
            g = Game(multidimensional=True)
            states[j, :, :, :] = g.board

        states2 = np.expand_dims(states.copy(), axis=1)
        
        for t in count():
            actions = select_action(states, model1).tolist()
            actions2 = select_action(states2, model2).tolist()
            
            # update states
            for j in range(batch_size):
                if i*batch_size + j in finished_games:
                    continue
                g = Game(multidimensional=True)
                g.from_board(states[j, :, :, :].squeeze(), multidimensional=True)
                g.move(actions[j])
                states[j, :, :, :] = g.board

                won = g.check_win()
                lost = g.check_loss()

                if won or lost:
                    finished_games.add(i*batch_size+j)
                    high_scores[i*batch_size+j] = max_value(states[j, :, :, :])


            # updates states2
            for j in range(batch_size):
                if i*batch_size + j in finished_games2:
                    continue
                g = Game(multidimensional=True)
                g.from_board(states2[j, :, :, :, :].squeeze(), multidimensional=True)
                g.move(actions2[j])
                states2[j, 0, :, :, :] = g.board

                won = g.check_win()
                lost = g.check_loss()

                if won or lost:
                    finished_games2.add(i*batch_size+j)
                    high_scores2[i*batch_size+j] = max_value(states2[j, :, :, :, :])

            print("iter: {}, finished games: model1 = {}, model2 = {}".format(t, len(finished_games), len(finished_games2)), end='\r')
            nonlostgame = (len(finished_games) + len(finished_games2)) == 2 * (i+1) * batch_size
            if nonlostgame:
                break

    # data statistics
    print("------------------------------------------------------------")
    print("model 1: 2d CNN model, model 2: 3d CNN model")
    # mean
    print("mean:")
    print("model1: {}, model2: {}".format(high_scores.mean(), high_scores2.mean()))
    # std
    print("std:")
    print("model1: {}, model2: {}".format(high_scores.std(), high_scores2.std()))
    # max
    print("max:")
    print("model1: {}, model2: {}".format(high_scores.max(), high_scores2.max()))
    # min
    print("min:")
    print("model1: {}, model2: {}".format(high_scores.min(), high_scores.min()))
    # % at each value
    score_dict = np.zeros((11))
    score_dict2 = np.zeros((11))

    for s1, s2 in zip(high_scores.tolist(), high_scores2.tolist()):
        score_dict[int(np.log2(s1))] += 1
        score_dict2[int(np.log2(s2))] += 1

    print("distribution:")
    print("model1: ", score_dict.tolist())
    print("model2: ", score_dict2.tolist()) 

if __name__ == "__main__":
    # args = sys.argv
    # test_model(args[1])

    compare_model('Model/saved_models/2dpolicy_net.pt', 'Model/saved_models/3dpolicy_net.pt')

