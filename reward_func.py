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
    return (successful * 2) - 1

    used_indexes = set([x for index_pair in combined_indexes for x in index_pair])
    
    # if next value wins or loses the game
    if terminated: # won
        return 4096
    if truncated: # loss
        return -4096

    if not successful: # needs to change so that it accounts for the added value
        # move did nothing
        return -4096 # arbitrary value change at will
    
    reward = max(combined_values + [0])
    # reward = 0

    # for i in range(1, 3, 1): # techinically only needs to check one value because up and down will combine the same value, normally 4
    #     if (i < 2) == (dir < 2):
    #         continue
    #     # if i == dir: # if we want to implement checks in all 4 directions (for if you want to check if the movement creates new oppurtunities)
    #     #     continue


    #     temp_game = Game()
    #     temp_game.from_board(prev.copy())
    #     temp_success, temp_combined_values, temp_combined_indexes = temp_game.move(i)

    #     # for giving the current reward as the maximum reward
    #     directional_reward = max(temp_combined_values + [0])

    #     if directional_reward > abs(reward):
    #         reward = -1 * directional_reward
        
    #     # # for giving the current reward as the sum of the values combined minus the values that could have been combined
    #     # for index_pair in temp_combined_indexes:
    #     #     if index_pair[0] in used_indexes or index_pair[1] in used_indexes:
    #     #         reward += prev[index_pair[0][0], index_pair[0][1]] * 2 # getting rid of duplicate values

    #     # reward -= sum(temp_combined_values) # instead of doing this maybe just take the value of the one with the largest magnitude

    # # if not reward:
    # #     reward = -4

    return reward # maybe also include created oppurtunities