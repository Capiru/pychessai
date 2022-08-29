import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.contrib.alpha_zero as az
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

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # agent_id = [0|1] -> policy depends on episode ID
    # This way, we make sure that both policies sometimes play agent0
    # (start player) and sometimes agent1 (player to move 2nd).
    return "main" if episode.episode_id % 2 == agent_id else "random"

try:
    def env_creator(env_config):
        return PettingChessEnvFunc()
    env = PettingZooEnv_v2()
    try:
        env.get_state()
    except:
        raise BaseException()
    register_env('myEnv', lambda config: PettingZooEnv_v2())
    ModelCatalog.register_custom_model("my_torch_model", LeelaZero)
    ray.shutdown()
    ray.init(ignore_reinit_error=True,object_store_memory=3*10**9)

    import shutil

    CHECKPOINT_ROOT = "tmp/ppo/taxi"
    #shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = "/ray_results/"
    #shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
    #config = (az.AlphaZeroConfig().environment(env="myEnv").training(model={"custom_model": "my_torch_model"}))

    #agent.restore(CHECKPOINT_ROOT+"\checkpoint_000163\checkpoint-163")
    N_ITER = 2000
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"

    tune.run(
        LeelaZeroTrainer,
        stop={"training_iteration": 500},
        max_failures=0,
        config={
            "env": "myEnv",
            "num_workers": 1,
            "num_gpus": 1,
            "mcts_config": {
                "num_simulations": 300,
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
                "epsilon": 0.05
            },
            "multiagent": {
            # Initial policy map: Random and PPO. This will be expanded
            # to more policy snapshots taken from "main" against which "main"
            # will then play (instead of "random"). This is done in the
            # custom callback defined above (`SelfPlayCallback`).
            "policies": {
                # Our main policy, we'd like to optimize.
                "main": PolicySpec(config = {"mcts_config": {
                "num_simulations": 30,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
                "epsilon": 0.01}}),
                # An initial random opponent to play against.
                "random": PolicySpec(config = {"mcts_config": {
                "num_simulations": 2,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": False,
                "epsilon": 1}}),#policy_class=RandomLegalPolicy),
            },
            # Assign agent 0 and 1 randomly to the "main" policy or
            # to the opponent ("random" at first). Make sure (via episode_id)
            # that "main" always plays against "random" (and not against
            # another "main").
            "policy_mapping_fn": policy_mapping_fn,
            # Always just train the "main" policy.
            "policies_to_train": ["main"],
        },
            "ranked_rewards": {
                "enable": False,
            },
            "model": {
                "custom_model": "my_torch_model",
            },
        },
    )
except Exception as e:
    print("Error")
    print(e)
    ray.shutdown()