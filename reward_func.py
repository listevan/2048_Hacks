import numpy as np
from Game_Sim.game import Game

def get_reward(prev, dir, next, terminated, truncated, successful, combined_values, combined_indexes):
    """
        0: up
        1: down
        2: right
        3: left
    """

    # positive
    # did it combine and what value combined
    # did it move it so two values can be combined
    # negative
    # did it do nothing
    # did it move two things that could have been combined apart

    used_indexes = set([x for index_pair in combined_indexes for x in index_pair])
    print(used_indexes)
    
    # if next value wins or loses the game
    if terminated: # won
        return 2048*2
    if truncated: # loss
        return -2048*2

    if not successful: # needs to change so that it accounts for the added value
        # move did nothing
        return -2048*2 # arbitrary value change at will
    
    reward = sum(combined_values)

    for i in range(1, 3, 1): # techinically only needs to check one value because up and down will combine the same value, normally 4
        if (i < 2) == (dir < 2):
            continue
        # if i == dir:
        #     continue
        print(i)
        temp_game = Game()
        temp_game.from_board(prev.copy())
        temp_success, temp_combined_values, temp_combined_indexes = temp_game.move(i)
        
        for index_pair in temp_combined_indexes:
            print(index_pair)
            if index_pair[0] in used_indexes or index_pair[1] in used_indexes:
                reward += prev[index_pair[0][0], index_pair[0][1]] * 2 # getting rid of duplicate values

        reward -= sum(temp_combined_values)

    if not reward:
        reward = -4

    return reward # maybe also include created oppurtunities