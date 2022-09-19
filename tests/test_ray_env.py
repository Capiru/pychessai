import sys
sys.path.append("C:\\Users\\gabri\\Documents\\pychessai")

from gym_env import ChessEnv,PettingChessEnvFunc,PettingChessEnv,PettingZooEnv_v2
import chess as ch
from pettingzoo.classic.chess import chess_utils

from ray_utils.randomlegalpolicy import RandomLegalPolicy

env = PettingChessEnv()
env2 = PettingZooEnv_v2(env=env)
env.reset()

print(env2.observe())