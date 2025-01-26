import numpy as np
from Game_Sim.game import Game


def get_reward(
    prev,
    dir,
    next,
    terminated,
    truncated,
    successful,
    combined_values,
    combined_indexes,
):
    """
    0: up
    1: down
    2: right
    3: left
    """

    # return if there is a new record
    prev_game = Game()
    prev_game.from_board(prev, multidimensional=True)
    next_game = Game()
    next_game.from_board(next, multidimensional=True)
    print("prev_max is {}, next_max is {}".format(prev_game.max(), next_game.max()))
    return (next_game.max() > prev_game.max()) * next_game.max()
