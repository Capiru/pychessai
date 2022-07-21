import ray
import ray.rllib.agents.ppo as ppo
import os
import gym
import gym_chess
from gym_env import ChessEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import ray_model
from ray_model import CustomTorchModel

try:
    SELECT_ENV = 'ChessAlphaZero-v0'
    env = gym.make(SELECT_ENV)
    def my_env(env_config):
        return gym.make(SELECT_ENV)
    register_env(SELECT_ENV, my_env)
    ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    import shutil

    CHECKPOINT_ROOT = "tmp/ppo/taxi"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = "/ray_results/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)



    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["framework"] = "torch"
    config["model"] = {"custom_model":"my_torch_model","custom_model_config":{}}
    agent = ppo.PPOTrainer(config=config, env=ChessEnv)

    N_ITER = 50
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    for n in range(N_ITER):
        result = agent.train()
        file_name = agent.save(CHECKPOINT_ROOT)

        print(s.format(
          n + 1,
          result["episode_reward_min"],
          result["episode_reward_mean"],
          result["episode_reward_max"],
          result["episode_len_mean"],
          file_name
        ))
except Exception as e:
    print("Error")
    print(e)
    ray.shutdown()