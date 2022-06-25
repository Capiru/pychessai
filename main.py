from config import CFG
from agents.LeelaZero import LeelaZeroAgent
from agents.RandomAgent import RandomAgent
from match import experiments
import chess as ch

board = ch.Board()
agent_one = LeelaZeroAgent(n_simulations = 10)
agent_one.value_model.to(CFG.DEVICE)
agent_one.value_model.get_board_evaluation(board)
agent_two = RandomAgent()
experiments(agent_one,agent_one,n= 20,is_update_elo=False,save_match_tensor=False)