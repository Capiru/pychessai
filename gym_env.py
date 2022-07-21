import gym
import gym_chess
import chess as ch
import numpy as np
from match import *
from move_choice import *
from policy import is_move_legal

class ChessEnv(gym.Env):
    def __init__(self,env_config):
        self.action_space = gym.spaces.Discrete(64)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(19,8,8), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(8*8*73)
        self.done = False
        self.info = {}
        self.reward = 0.0

    def _observe(self):
        return get_board_as_tensor(self.board,self.board.turn).cpu().numpy()

    def reset(self):
        self.board = ch.Board()
        self.reward = 0
        self.done = False
        return self._observe()
    
    def _reward(self,condition):
        if condition:
            self.reward = 1
        elif condition is None:
            self.reward = 0
        else:
            self.reward = -1
        return self.reward

    def _done(self):
        if self.board.is_game_over():
            self.done = True
        return self.done

    def is_action_legal(self,action):
        return is_move_legal(action,self.board)

    def push(self,action):
        move = map_policy_to_move(action,self.board)
        self.board.push(move)

    def step(self,action):
        if self.is_action_legal(action):
            self.push(action)
            self.done = self._done()
            if self.done:
                winner = self.board.outcome.winner
            else:
                winner = None
        else:
            winner = False
            self.done = True
        return self._observe(),self._reward(winner),self.done,self.info

