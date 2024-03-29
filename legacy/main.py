from config import CFG
from agents.LeelaZero import LeelaZeroAgent
from agents.RandomAgent import RandomAgent
from agents.NegaMaxAgents import NegaMaxAgent
from agents.MinimaxAgents import MinimaxPruningSimplePolicyAgent
from match import experiments
import chess as ch
from training import *
from test import *
 
import torch

#board = ch.Board('rn2kb1r/p1qBpppp/2pP3N/1p6/Q2p3P/2P3pR/PP1nPP2/R1B1KBN1 b Qkq - 2 8')
agent_one = LeelaZeroAgent(n_simulations = 50,res_blocks = 10,filters = 194)
agent_one.value_model.to(CFG.DEVICE)
#agent_two = NegaMaxAgent(depth = 2,save_policy= True)
agent_two = MinimaxPruningSimplePolicyAgent(depth = 2,save_policy= True)
rangent = RandomAgent()
# CFG.RANDOM_START = 0
# find_checkmate_in_1(agent_one)
# # print("passed test ckmt 1")
# # find_checkmate_in_2(agent_one)
# # print("passed test ckmt 2")
# #print(training_test(agent_one))
# find_hanging_piece(agent_one)
#experiments(agent_one,rangent,n = 5,save_match_tensor=False)
#print(match(agent_one,agent_two,save_tensor = True,is_player_one = True))
#print(list(board.legal_moves))
#print(agent_one.choose_move(board))
self_play(agent_one,base_agent = None,play_batch_size = 64,n_accumulate=2)
