import numpy as np
def check_direction(board, i, j, dir):
    """
        0: up
        1: down
        2: right
        3: left
    """ 

    # checking vertical
    if dir <= 1:
        if dir == 0:
            next_values = range(j-1, -1, -1)
        elif dir == 1:
            next_values = range(j+1, 4, 1)
        for jj in next_values:
            if board[i][jj] == board[i][j]:
                return True
            elif board[i][jj] != 0:
                break
    else:
        if dir == 2:
            next_values = range(i+1, 4, 1)
        elif dir == 3:
            next_values = range(i-1, -1, -1)
        for ii in next_values:
            if board[ii][j] == board[i][j]:
                return True
            elif board[ii][j] != 0:
                break   
    return False

def check_combinations(board, i, j):
    # checks if the value at the current position could have been combined in any direction
    """
        0: up
        1: down
        2: right
        3: left
    """
    ans = [0] * 4 # for each direction 

    if board[i][j] == 0:
        return [0, 0]
    
    for dir in range(4):
        ans[i] = check_direction(board, i, j, dir)

    return [ans[0]+ ans[1], ans[2]+ans[3]]

def get_reward(prev, next, terminated, truncated):
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

    if (prev == next).all(): # needs to change so that it accounts for the added value
        # move did nothing
        return -2048*2 # arbitrary value change at will
    
    reward = 0
    
    for i in range (4):
        for j in range (4):
            # could this have been combined in previous
            prev_check = check_combinations(prev, i, j)
            # can this be combined in next
            next_check = check_combinations(next, i, j)
            
            # Vertical TS
            if prev_check[0] > 0: # if previously could be combined in vertical direction
                # if combined
                if np.sum(next, axis=1)[j] > np.sum(prev, axis=1)[j] or (next[:, j]==0).sum() > (prev[:, ]==0).sum():
                    reward += 2*prev[i][j]
                # if not combined
                elif prev_check[0] == next_check[0]:
                    reward -= prev[i][j]
                # if removed pairing
                else:
                    reward -= 2*prev[i][j]
            elif prev_check[0] == 0:
                # if can now be combined
                if next_check[0]:
                    reward += next[i][j]

            # Horizontal TS
            if prev_check[1] > 0: # if previously could be combined in vertical direction
                # if combined
                if np.sum(next, axis=0)[i] > np.sum(prev, axis=0)[i] or (next[i, :]==0).sum() > (prev[i, :]==0).sum():
                    reward += 2*prev[i][j]
                # if not combined
                elif prev_check[1] == next_check[1]:
                    reward -= prev[i][j]
                # if removed pairing
                else:
                    reward -= 2*prev[i][j]
            elif prev_check[1] == 0:
                # if can now be combined
                if next_check[1]:
                    reward += next[i][j]
    return reward

