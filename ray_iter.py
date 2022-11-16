import ray
import ray.rllib.agents.ppo as ppo
import os

import ray.tune as tune

import ray_utils.leelazero_model
from ray_utils.leelazero_model import LeelaZero

import numpy as np
from functools import wraps
import chess as ch

from ray_utils.leelazero_trainer import LeelaZeroTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray_utils.randomlegalpolicy import RandomLegalPolicy
from ray_utils.leelazero_policy import LeelaZeroPolicy

from helper import *

setup_leela()
ray.shutdown()
ray.init(ignore_reinit_error=True,object_store_memory=3*10**9)
mcts_config = {"mcts_config": {
                "num_simulations": 100,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": False,
                "argmax_child_value":True,
                "puct_coefficient":np.sqrt(2),
                "epsilon": 0.05,
                "turn_based_flip":True}}
rand_config = {"mcts_config": {
                "num_simulations": 10,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": False,
                "argmax_child_value":True,
                "puct_coefficient":np.sqrt(2),
                "epsilon": 0.99,
                "turn_based_flip":True}}
#az_config = (az.AlphaZeroConfig().environment(env="myEnv").training(model={"custom_model": "my_torch_model"}))
config={    
            "env": 'ChessMultiAgent',
            "num_workers": 4,
            "num_gpus": 1,
            "train_batch_size": 512,
            "rollout_fragment_length": 200,
            "horizon": 300,
            "multiagent": {
            # Initial policy map: Random and PPO. This will be expanded
            # to more policy snapshots taken from "main" against which "main"
            # will then play (instead of "random"). This is done in the
            # custom callback defined above (`SelfPlayCallback`).
            "policies": {
                # Our main policy, we'd like to optimize.
                "main": PolicySpec(config = mcts_config),
                #"main_dplg": PolicySpec(config = mcts_config),
                # An initial random opponent to play against.
                #"random": az(config = (az.AlphaZeroConfig().environment(env="myEnv").training(model={"custom_model": "my_torch_model"})))
                "random": PolicySpec(config = rand_config),#policy_class=RandomLegalPolicy),
            },
            # Assign agent 0 and 1 randomly to the "main" policy or
            # to the opponent ("random" at first). Make sure (via episode_id)
            # that "main" always plays against "random" (and not against
            # another "main").
            "policy_mapping_fn": competition_mapping_fn,
            # Always just train the "main" policy.
            "policies_to_train": ["main"],
        },
            "ranked_rewards": {
                "enable": False,
            },
            "model": {
                "custom_model": "LeelaZero",
            },
        }
LeelaTrainer = LeelaZeroTrainer(config = config)

import shutil

CHECKPOINT_ROOT = "G:\\Meine Ablage\\projects\\chessai\\tmp\\ppo\\pychessai"
#shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = "/ray_results/"

#shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

checkpoint = return_latest_checkpoint(dir_path=CHECKPOINT_ROOT)
if checkpoint != "":
    LeelaTrainer.restore(CHECKPOINT_ROOT+"\\"+checkpoint)

N_ITER = 2000
s = "{:3d} reward {:6.5f}/{:6.5f}/{:6.5f}/{:6.2f} len {:6.2f}"
print("it reward     ttl/   pll/   vll/  rew   len avg")

file_name = ""
for n in range(N_ITER):
    result = LeelaTrainer.train()
    if (n + 1) % 5 == 0:
        file_name = LeelaTrainer.save(CHECKPOINT_ROOT)
        print("Checkpoint saved ",file_name)
        LeelaTrainer.export_policy_model(CHECKPOINT_ROOT+"\\bestmodel","main")
        #LeelaTrainer.import_policy_model_from_h5(CHECKPOINT_ROOT,"main_dplg")
    print(s.format(
        n + 1,
        result["info"]["learner"]["main"]["learner_stats"]['total_loss'],
        result["info"]["learner"]["main"]["learner_stats"]['policy_loss'],
        result["info"]["learner"]["main"]["learner_stats"]['value_loss'],
        result["policy_reward_mean"]["main"],
        result["episode_len_mean"],
    ))
    eval_ = LeelaTrainer.eval()
   # if elo > max_elo: LeelaTrainer.add_policy("new_policy") LeelaTrainer.remove_policy("random")


