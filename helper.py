import os
from ray_utils.leelazero_model import LeelaZero
from ray_utils.leelazero_trainer import LeelaZeroTrainer
from ray.rllib.env import PettingZooEnv
from gym_env import ChessEnv,PettingChessEnvFunc,PettingChessEnv,PettingZooEnv_v2
from ray.tune.registry import register_env, register_trainable
from ray.rllib.models import ModelCatalog

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # agent_id = [0|1] -> policy depends on episode ID
    # This way, we make sure that both policies sometimes play agent0
    # (start player) and sometimes agent1 (player to move 2nd).
    return "main" if int(agent_id.split("_")[-1]) % 2 == 0 else "random"

def competition_mapping_fn(agent_id,episode,worker,**kwargs):
    if int(episode.episode_id) <= 100:
        if int(episode.episode_id)%2 == 0:
            policy = "main" if int(agent_id.split("_")[-1]) % 2 == 0 else "random"
        else:
            policy = "random" if int(agent_id.split("_")[-1]) % 2 == 0 else "main"
    else:
        if int(episode.episode_id)%2 == 0:
            policy = "main" if int(agent_id.split("_")[-1]) % 2 == 0 else "main"
        else:
            policy = "main" if int(agent_id.split("_")[-1]) % 2 == 0 else "main"
    return policy

def register_chess_env():
    def env_creator(env_config):
        return PettingChessEnvFunc()
    env = PettingZooEnv_v2()
    try:
        env.get_state()
    except:
        raise BaseException()
    register_env('ChessMultiAgent', lambda config: PettingZooEnv_v2())

def register_leela_model():
    ModelCatalog.register_custom_model("LeelaZero", LeelaZero)

def register_leela_trainable():
    register_trainable("LeelaTrainer",LeelaZeroTrainer)

def return_latest_checkpoint(dir_path):
    cpt_list = os.listdir(dir_path)
    ordered_checkpoint = {}
    max_score = -1
    file_name = ""
    for f in cpt_list:
        try:
            score = int(f.split("_")[-1])
            if score > max_score:
                max_score = score
                file_name = f
        except Exception:
            pass
    return file_name

def setup_leela():
    register_chess_env()
    register_leela_model()
    register_leela_trainable()