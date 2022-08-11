from gym_env import ChessEnv,PettingChessEnvFunc,PettingChessEnv,PettingZooEnv_v2
import chess as ch
from pettingzoo.classic.chess import chess_utils
from search.search import MonteCarloSearchNode
from agents.LeelaZero import LeelaZeroAgent
from config import CFG
from ray_utils.mcts import Node, RootParentNode, MCTS
from ray_utils.leelazero_model import LeelaZero
from ray_utils.leelazero_trainer import LeelaZeroTrainer

agent = LeelaZeroAgent()
agent.value_model.to(CFG.DEVICE)

env = PettingChessEnv()
env2 = PettingZooEnv_v2()
env.reset()
chess_utils.make_move_mapping("e2e4")
action = chess_utils.moves_to_actions["e2e4"]
env.step(action)

chess_utils.make_move_mapping("e2e4")
action2 = chess_utils.moves_to_actions["e2e4"]
env.step(action2)

chess_utils.make_move_mapping("d1h5")
action = chess_utils.moves_to_actions["d1h5"]
env.step(action)

chess_utils.make_move_mapping("a2a3")
action2 = chess_utils.moves_to_actions["a2a3"]
env.step(action2)

chess_utils.make_move_mapping("f1c4")
action = chess_utils.moves_to_actions["f1c4"]
env.step(action)

chess_utils.make_move_mapping("a3a4")
action2 = chess_utils.moves_to_actions["a3a4"]
env.step(action2)

parent_node = MonteCarloSearchNode(agent,None,not True,env.board)
score,move = parent_node.search(n_simulations = 100)
print(score,move)

chess_utils.make_move_mapping("h5f7")
action = chess_utils.moves_to_actions["h5f7"]
env.step(action)

print(env.board)
print(env.rewards)
print(env.dones)