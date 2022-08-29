from gym_env import ChessEnv,PettingChessEnvFunc,PettingChessEnv,PettingZooEnv_v2
import chess as ch
from pettingzoo.classic.chess import chess_utils
from search.search import MonteCarloSearchNode
from agents.LeelaZero import LeelaZeroAgent
from config import CFG
from ray_utils.mcts import Node, RootParentNode, MCTS
from ray_utils.leelazero_model import LeelaZero
from ray_utils.leelazero_trainer import LeelaZeroTrainer
import gym.spaces as spaces
import numpy as np
import torch

agent = LeelaZeroAgent()
agent.value_model.to(CFG.DEVICE)

env = PettingChessEnv()
env2 = PettingZooEnv_v2(env=env)
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

print(env.board.fen())

parent_node = MonteCarloSearchNode(agent,None,not True,env.board)
score,move = parent_node.search(n_simulations = 100)
print(score,move)

def test_mcts(env,right_move):
    model = LeelaZero(obs_space = spaces.Box(
                            low=0, high=1, shape=(8, 8, 111), dtype=bool
                        ),action_space = spaces.Box(
                            low=0, high=1, shape=(4672,), dtype=np.int8
                        ),num_outputs = 1,model_config = {},name = "LeelaZero")
    mcts = MCTS(model, {"temperature":1.0,"dirichlet_epsilon":0.25,"dirichlet_noise":0.25,"num_simulations":2000,"argmax_tree_policy":True,"argmax_child_value":True,"add_dirichlet_noise":True,
            "puct_coefficient":np.sqrt(2),"epsilon":0,"turn_based_flip":True})
    tree_node = Node(
                            state=env.get_state(),
                            obs=env.observe(),
                            reward=0,
                            done=False,
                            action=None,
                            parent=RootParentNode(env=env),
                            mcts=mcts,
                        )
    mcts_policy, action4, tree_node = mcts.compute_action(tree_node)
    move = chess_utils.actions_to_moves[action4]
    print(action4,move,chess_utils.mirror_move(ch.Move.from_uci(move)),mcts_policy)
    ### 2869^
    try:
        assert move == right_move
    except AssertionError as e:
        print("Incorrect Move: ",move,"correct move:", right_move)
        raise AssertionError()



def test_mcts_checkmate_in_2():

    env = PettingChessEnv()
    env2 = PettingZooEnv_v2(env=env)
    env.board = ch.Board("6k1/pp3pp1/2B5/3N2bp/8/1Q1P1p2/PPP2PbK/R1B1r3 b - - 0 22")
    env.agent_selection = (
                env._agent_selector.next()
            )

    test_mcts(env2, right_move="e8h8")

def test_mcts_checkmate_in_1():
    env = PettingChessEnv()
    env2 = PettingZooEnv_v2(env=env)
    env.board = ch.Board("rnbqkbnr/1ppp1ppp/8/p3p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4")

    test_mcts(env2, right_move="h5f7")
    
test_mcts_checkmate_in_1()
print("Completed checkmate in 1 test")
test_mcts_checkmate_in_2()
print("Completed checkmate in 2 test")

print(env.board)
print(env.rewards)
print(env.dones)
