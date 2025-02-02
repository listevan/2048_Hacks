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

    prev_game = Game()
    prev_game.from_board(prev, multidimensional=True)
    next_game = Game()
    next_game.from_board(next, multidimensional=True)
    strategy = 0
    """
    List of Strategies:
    Highest number in the corner, start by implementing just for top corner
    Merging smaller values, may lead to local minima
    Avoid flat boards
    """

    max_layer = np.log(next_game.max())
    if not (
        prev_game.board[0, 0] == next_game.max()
        or prev_game.board[0, 3] == next_game.max()
    ) and (
        next_game.board[0, 0] == next_game.max()
        or next_game.board[0, 3] == next_game.max()
    ):  # max in corner
        strategy += 0.2
    for combined_value in combined_values:
        strategy += (
            np.log(combined_value) / max_layer
        )  # reduce the usefulness of values already achieved
    for i in [2, 3]:
        temp_next_game = Game()
        temp_next_game.from_board(next, multidimensional=True)
        successful, _, _ = temp_next_game.move(i)
        if not successful:  # if no lateral movement is possible
            successful -= 1

    if next_game.max() > prev_game.max():
        return int(max_layer)
    elif strategy > 0:
        return strategy
    else:
        return -0.1 + strategy
