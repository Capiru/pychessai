### Simulator Agent Script
# The purpose of the simulator agent is to:
# 1) Produce tensor files from self play matches
# 2) Decide which model pools to use for self-play
# 3) Update model pools when needed
import numpy as np
import torch
from match import *
import os

def self_play_simulator(model_pool,n_matches):
    ### Take model_pool and number of matches as input, produce n_matches tensors as output in the matches dir
    random_idx = np.random.randint(0,len(model_pool)-1,2)
    agent_one = model_pool[random_idx[0]]
    agent_two = model_pool[random_idx[1]]
    experiments(agent_one,agent_two,n=n_matches,is_update_elo=True,start_from_opening = True,progress_bar = False,save_match_tensor = True)
    return None

def choose_model_pool(model_dir):
    model_list = {}
    model_elos = []
    max_elo = 0
    model_pool = []
    for file in os.listdir(model_dir):
        continue
    return model_pool

def update_model_pool(model_dir,model_pool):
    new_model_pool = []
    return new_model_pool

while True:
    break