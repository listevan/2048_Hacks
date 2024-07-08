"""
    Rules:
       starts with two values (only twos (%) and fours (%) can spawn)
       each time a player moves the pieces a two (%) or a four (%) spawn
       moving pieces adds two consecutive same values in that row (when player moves left or right) or column (when player moves up or down) (if three values which combine)
"""

import numpy as np

class Game:
    def __init__(self):
        self._board = np.zeros([4, 4])
        self.gen_values()
        self.gen_values()

    def display(self):
        print(self._board)

    def gen_values(self):
        """Generates one value"""
        emptys = self.get_empty_cells()
        if (not len(emptys)):
            return
        chosen = np.random.choice(emptys, size=1, replace=False)
        i1 = chosen[0] // 4
        j1 = chosen[0] % 4
        # i2 = chosen[1] // 4
        # j2 = chosen[1] % 4
        # print("gen'd at", i1, j1)
        self._board[i1][j1] = np.random.choice([2, 4], size=1)[0]
        # self._board[i2][j2] = np.random.choice([2, 4], size=1)[0]

    def move(self, direction):
        """
            0: up
            1: down
            2: right
            3: left
        """
        # do movement, generate value
        if direction == 0:
            self.moveVertical(1) 
        elif direction == 1:
            self.moveVertical(-1)
        elif direction == 2:
            self.moveHorizontal(-1)
        elif direction == 3:
            self.moveHorizontal(1)


        # self.display()
        
        self.gen_values()

    def moveVertical(self, direction):
        """1 for up, -1 for down"""
        # Do column first to go down the column
        for c in range(4):
            prev_empty = []
            last_value = None
            loop_range = range(4) if direction == 1 else reversed(range(4))
            for r in loop_range:
                if self._board[r][c] == 0:
                    prev_empty.append((r, c))
                    continue

                if last_value and self._board[last_value[0]][last_value[1]] == self._board[r][c]:
                    self._board[last_value[0]][last_value[1]] *= 2
                    self._board[r][c] = 0
                    last_value = None

                last_value = (r, c)


                if len(prev_empty) == 0:
                    continue 

                empty_cell = prev_empty.pop(0)

                self._board[empty_cell[0]][empty_cell[1]] = self._board[r][c]
                last_value = empty_cell
                self._board[r][c] = 0
                prev_empty.append((r, c))

    def moveHorizontal(self, direction):
        """1 for left, -1 for right"""
        # Do column first to go down the column
        for r in range(4):
            prev_empty = []
            last_value = None
            loop_range = range(4) if direction == 1 else reversed(range(4))
            for c in loop_range:
                if self._board[r][c] == 0:
                    prev_empty.append((r, c))
                    continue

                if last_value and self._board[last_value[0]][last_value[1]] == self._board[r][c]:
                    self._board[last_value[0]][last_value[1]] *= 2
                    self._board[r][c] = 0
                    last_value = None

                last_value = (r, c)

                if len(prev_empty) == 0:
                    continue 

                empty_cell = prev_empty.pop(0)
                self._board[empty_cell[0]][empty_cell[1]] = self._board[r][c]
                last_value = empty_cell
                self._board[r][c] = 0
                prev_empty.append((r, c))

    @property
    def board(self):
        """Returns the current state of the board."""
        return self._board

    def check_loss(self) -> bool:
        """Returns True if there's no moves to be made, False otherwise."""
        for i in range(4):
            for j in range(4):
                left = (i, j-1)
                right = (i, j+1)
                up = (i-1, j)
                down = (i+1, j)

                if self._in_bounds(*left):
                    if self._board[i][j] == self._board[left[0]][left[1]]:
                        return False
                if self._in_bounds(*right):
                    if self._board[i][j] == self._board[right[0]][right[1]]:
                        return False
                if self._in_bounds(*up):
                    if self._board[i][j] == self._board[up[0]][up[1]]:
                        return False
                if self._in_bounds(*down):
                    if self._board[i][j] == self._board[down[0]][down[1]]:
                        return False
        return True

    def _in_bounds(self, r, c):
        return (r >= 0 and c >= 0) and (r <= 3 and c <= 3)

    def check_win(self) -> bool:
        """Returns True if there's a 2048 value in the board, False otherwise."""
        for i in range(4):
            for j in range(4):
                if self._board[i][j] == 2048:
                    return True
        return False

    def get_occupied_cells(self):
        """Returns a list of occupied cells as tuples"""
        res = []
        for i in range(4):
            for j in range(4):
                if self._board[i][j] != 0:
                    res.append((i, j))
        return res

    def get_empty_cells(self):
        """Returns a list of the empty cells."""
        res = [] 
        for i in range(4):
            for j in range(4):
                if self._board[i][j] == 0:
                    res.append(i * 4 + j)
        return res