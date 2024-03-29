import logging
from typing import Type

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm import with_common_config
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.replay_ops import (
    SimpleReplayBuffer,
    Replay,
    StoreToReplayBuffer,
    WaitUntilTimestepsElapsed,
)
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TrainerConfigDict
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.util.iter import LocalIterator

from ray_utils.leelazero_policy import LeelaZeroPolicy
from ray_utils.legacy_leela_policy import LeelaZeroLegacyPolicy
from ray_utils.mcts import MCTS

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class LeelaZeroDefaultCallbacks(DefaultCallbacks):
    """LeelaZero callbacks.
    If you use custom callbacks, you must extend this class and call super()
    for on_episode_start.
    """
    def __init__(self):
        super().__init__()

    @override(DefaultCallbacks)
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        # save env state when an episode starts
        env = base_env.get_sub_environments()[0]
        state = env.get_state()
        episode.user_data["initial_state"] = state
        episode.user_data["current_state"] = [state]

    @override(DefaultCallbacks)
    def on_episode_step(
        self, worker, base_env, policies,
        episode, **kwargs) -> None:
        env = base_env.get_sub_environments()[0]
        state = env.get_state()
        episode.user_data["current_state"].append(state)

    @override(DefaultCallbacks)
    def on_episode_end(self,worker,base_env,policies,episode,**kwargs):
        env = base_env.get_sub_environments()[0]
        if env.env.board.outcome():
            winner = env.env.board.outcome().winner
        else:
            winner = "Draw"
        logging.info("Game Over:"+str(winner)+"Reward:"+str(policies)+str(episode.agent_rewards))





# fmt: off
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # Size of batches collected from each worker
    "rollout_fragment_length": 200,
    # Number of timesteps collected for each SGD round
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended)
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # IN case a buffer optimizer is used
    "learning_starts": 1000,
    # Size of the replay buffer in batches (not timesteps!).
    "buffer_size": 1000,
    # Stepsize of SGD
    "lr": 5e-5,
    # Learning rate schedule
    "lr_schedule": None,
    # Share layers for value function. If you set this to True, it"s important
    # to tune vf_loss_coeff.
    "vf_share_layers": False,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "complete_episodes",
    # Which observation filter to apply to the observation
    "observation_filter": "NoFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This does
    # not support minibatches.
    "simple_optimizer": True,

    # === MCTS ===
    "mcts_config": {
        "puct_coefficient": 1.0,
        "num_simulations": 30,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": False,
        "add_dirichlet_noise": True,
        "epsilon": 0.05,
        "turn_based_flip":True,
        "argmax_child_value":True,
    },

    # === Ranked Rewards ===
    # implement the ranked reward (r2) algorithm
    # from: https://arxiv.org/pdf/1807.01672.pdf
    "ranked_rewards": {
        "enable": False,
        "percentile": 75,
        "buffer_max_length": 1000,
        # add rewards obtained from random policy to
        # "warm start" the buffer
        "initialize_buffer": True,
        "num_init_rewards": 100,
    },

    # === Evaluation ===
    # Extra configuration that disables exploration.
    "evaluation_config": {
        "mcts_config": {
            "argmax_tree_policy": True,
            "add_dirichlet_noise": False,
        },
    },

    # === Callbacks ===
    "callbacks": LeelaZeroDefaultCallbacks,

    "framework": "torch",  # Only PyTorch supported so far.
})


# __sphinx_doc_end__
# fmt: on


def leela_zero_loss(policy, model, dist_class, train_batch):
    # get inputs unflattened inputs
    input_dict = restore_original_dimensions(
        train_batch["obs"], policy.observation_space, "torch"
    )
    # forward pass in model
    model_out = model.forward(input_dict, None, [1])
    logits, _ = model_out
    values = model.value_function()
    logits, values = torch.squeeze(logits), torch.squeeze(values)
    priors = nn.Softmax(dim=-1)(logits)
    # compute actor and critic losses
    policy_loss = torch.mean(
        -torch.sum(train_batch["mcts_policies"] * torch.log(priors), dim=-1)
    )
    value_loss = torch.mean(torch.pow(values - train_batch["value_label"], 2))
    # compute total loss
    total_loss = (policy_loss + value_loss) / 2
    return total_loss, policy_loss, value_loss


class LeelaZeroPolicyWrapperClass(LeelaZeroPolicy):
    def __init__(self, obs_space, action_space, config):
        model = ModelCatalog.get_model_v2(
            obs_space, action_space, action_space.n, config["model"], "torch"
        )
        env_creator = _global_registry.get(ENV_CREATOR, config["env"])
        # if config["ranked_rewards"]["enable"]:
        #     # if r2 is enabled, tne env is wrapped to include a rewards buffer
        #     # used to normalize rewards
        #     env_cls = get_r2_env_wrapper(env_creator, config["ranked_rewards"])

        #     # the wrapped env is used only in the mcts, not in the
        #     # rollout workers
        #     def _env_creator():
        #         return env_cls(config["env_config"])

        # else:

        def _env_creator():
            return env_creator(config["env_config"])

        def mcts_creator():
            return MCTS(model, config["mcts_config"])

        super().__init__(
            obs_space,
            action_space,
            config,
            model,
            leela_zero_loss,
            TorchCategorical,
            mcts_creator,
            _env_creator,
        )


class LeelaZeroTrainer(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(Algorithm)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return LeelaZeroPolicyWrapperClass


    # @staticmethod
    # @override(Algorithm)
    # def execution_plan(
    #     workers: WorkerSet, config: TrainerConfigDict, **kwargs
    # ) -> LocalIterator[dict]:
    #     assert (
    #         len(kwargs) == 0
    #     ), "Alpha zero execution_plan does NOT take any additional parameters"

    #     rollouts = ParallelRollouts(workers, mode="bulk_sync")

    #     if config["simple_optimizer"]:
    #         train_op = rollouts.combine(
    #             ConcatBatches(
    #                 min_batch_size=config["train_batch_size"],
    #                 count_steps_by=config["multiagent"]["count_steps_by"],
    #             )
    #         ).for_each(TrainOneStep(workers, num_sgd_iter=config["num_sgd_iter"]))
    #     else:
    #         replay_buffer = SimpleReplayBuffer(config["buffer_size"])

    #         store_op = rollouts.for_each(
    #             StoreToReplayBuffer(local_buffer=replay_buffer)
    #         )

    #         replay_op = (
    #             Replay(local_buffer=replay_buffer)
    #             .filter(WaitUntilTimestepsElapsed(config["learning_starts"]))
    #             .combine(
    #                 ConcatBatches(
    #                     min_batch_size=config["train_batch_size"],
    #                     count_steps_by=config["multiagent"]["count_steps_by"],
    #                 )
    #             )
    #             .for_each(TrainOneStep(workers, num_sgd_iter=config["num_sgd_iter"]))
    #         )

    #         train_op = Concurrently(
    #             [store_op, replay_op], mode="round_robin", output_indexes=[1]
    #         )

    #     return StandardMetricsReporting(train_op, workers, config)