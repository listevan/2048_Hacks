"""
    Rules:
       starts with two values (only twos (%) and fours (%) can spawn)
       each time a player moves the pieces a two (%) or a four (%) spawn
       moving pieces adds two consecutive same values in that row (when player moves left or right) or column (when player moves up or down) (if three values which combine)
"""

import numpy as np

class game:
    def __init__ (self):
        self.board = np.zero(4, 4)
        # generate two values

    def move (self, direction):
        """
            1: up
            2: right
            3: down
            4: left
        """
        # do movement, generate value
    
    def getBoard (self):
        return self.board