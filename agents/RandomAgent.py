import chess as ch
class RandomAgent(object):
    def __init__(self):
        self.elo = 400
    def choose_move(self,fen):
        board = ch.Board(fen)
        legal_moves = list(board.legal_moves)
        return random_choice(legal_moves,None,1)
