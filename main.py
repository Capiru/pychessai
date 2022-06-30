from config import CFG
from agents.LeelaZero import LeelaZeroAgent
from agents.RandomAgent import RandomAgent
from match import experiments
import chess as ch
from training import *

# board = ch.Board("7k/1p1R4/6KP/8/4P3/2P3P1/8/qr6 w - - 0 50")
agent_one = LeelaZeroAgent(n_simulations = 20)
agent_one.value_model.to(CFG.DEVICE)
self_play(agent_one,base_agent = None,play_batch_size = 64,n_accumulate=30)
# agent_one.value_model.get_board_evaluation(board)
# agent_one.is_white = True
# print(agent_one.choose_move(board))
# agent_two = RandomAgent()
#experiments(agent_one,agent_two,n= 3,is_update_elo=False,save_match_tensor=False)