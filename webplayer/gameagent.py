import chess

from game import Game
from agents.LeelaZero import LeelaZeroAgent
import torch


class GameAgent(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self,
                 agent,
                 player_color=Game.WHITE,
                 board=None,
                 date=None):
        super().__init__(board=board, player_color=player_color, date=date)

        self.agent = LeelaZeroAgent(n_simulations = 300,res_blocks = 10,filters = 194)
        self.agent.is_white = not self.player_color
        self.agent.color = not self.player_color
        state_dict = torch.load('models/value_model.pth',map_location=torch.device('cpu'))
        self.agent.value_model.load_state_dict(state_dict)

    def move(self, movement):
        """ Makes a move. If it's not your turn, the agent will play and if
        the move is illegal, it will be ignored.

        Params:
            movement: str, Movement in UCI notation (f2f3, g8f6...)
        """
        made_movement = False
        # If agent moves first (whites and first move)
        if self.agent.color and len(self.board.move_stack) == 0:
            agents_best_move = self.agent.choose_move(self.board)
            self.board.push(agents_best_move[0])
            made_movement = True
        else:
            made_movement = super().move(movement)
            if made_movement and self.get_result() is None:
                agents_best_move = self.agent.choose_move(self.board)
                self.board.push(agents_best_move[0])
        return made_movement

    def get_copy(self):
        return GameAgent(board=self.board.copy(), agent=self.agent,
                         player_color=self.player_color)

    def tearup(self):
        """ Free resources."""
        del(self.agent)
