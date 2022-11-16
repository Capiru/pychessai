import ray
import ray.rllib.agents.ppo as ppo
import os
import gym
import gym_chess
from gym_env import ChessEnv,PettingChessEnvFunc,PettingChessEnv,PettingZooEnv_v2
from ray.tune.registry import register_env
import ray.tune as tune
from ray.rllib.models import ModelCatalog
import ray_utils.leelazero_model
from ray_utils.leelazero_model import LeelaZero
from pettingzoo.classic import rps_v2,chess_v5
from ray.rllib.env import PettingZooEnv
import numpy as np
from functools import wraps
import chess as ch

from ray_utils.leelazero_trainer import LeelaZeroTrainer
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray_utils.randomlegalpolicy import RandomLegalPolicy
from ray_utils.leelazero_policy import LeelaZeroPolicy

from helper import *

try:
    setup_leela()
    ray.shutdown()
    ray.init()

    import shutil

    CHECKPOINT_ROOT = "tmp/ppo/taxi"
    #shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = "/ray_results/"
    #shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
    #config = (az.AlphaZeroConfig().environment(env="myEnv").training(model={"custom_model": "my_torch_model"}))

    #agent.restore(CHECKPOINT_ROOT+"\checkpoint_000163\checkpoint-163")
    N_ITER = 2000
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"
    mcts_config = {"mcts_config": {
                "num_simulations": 2000,
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
                "argmax_child_value":True,
                "puct_coefficient":np.sqrt(2),
                "epsilon": 0.00,
                "turn_based_flip":True}}

    #LeelaZeroTrainer.train()
    try:
        tuner = tune.Tuner.restore("G:\\Meine Ablage\\projects\\chessai\\ray_results\\pychessai").fit()
    except Exception as e:
        print(e)
    
    tune.run(
        "LeelaTrainer",
        name = "pychessai",
        stop={"training_iteration": 1000},
        checkpoint_freq=100,
        keep_checkpoints_num= 5,
        checkpoint_at_end=True,
        sync_config=tune.SyncConfig(
            syncer = None#upload_dir="G:\\Meine Ablage\\projects\\chessai\\ray_results"
        ),
        max_failures=0,
        local_dir="G:\\Meine Ablage\\projects\\chessai\\ray_results",
        config={
            "env": 'ChessMultiAgent',
            "num_workers": 6,
            "num_gpus": 1,
            "train_batch_size": 2048,
            "rollout_fragment_length": 200,
            "horizon": 300,
            "multiagent": {
            "policies": {
                # Our main policy, we'd like to optimize.
                "main": PolicySpec(config = {"mcts_config": {
                "num_simulations": 100,
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
                "argmax_child_value":True,
                "puct_coefficient":np.sqrt(2),
                "epsilon": 0.05,
                "turn_based_flip":True}}),
                # An initial random opponent to play against.
                "random": PolicySpec(config = {"mcts_config": {
                "num_simulations": 5,
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
                "argmax_child_value":True,
                "epsilon": 1,
                "turn_based_flip":True}}),#policy_class=RandomLegalPolicy),
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
        },
    )
except Exception as e:
    print("Error")
    print(e)
    ray.shutdown()