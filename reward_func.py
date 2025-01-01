import numpy as np
from Game_Sim.game import Game

def get_reward(prev, dir, next, terminated, truncated, successful, combined_values):
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
    
    # if next value wins or loses the game
    if terminated: # won
        return 2048*2
    if truncated: # loss
        return -2048*2

    if not successful: # needs to change so that it accounts for the added value
        # move did nothing
        return -2048*2 # arbitrary value change at will
    
    reward = sum(combined_values)

    for i in range(4):
        if i == dir:
            continue
        temp_game = Game()
        temp_game.from_board(prev.copy())
        temp_success, temp_combined_values = temp_game.move(i)
        
        reward -= sum(temp_combined_values)

    if not reward:
        reward = -4

    return reward # maybe also include created oppurtunities