import torch.nn as nn
import torch
import ray
import gym
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.annotations import override

class CustomTorchModel(TorchModelV2,nn.Module):
    def __init__(self,obs_space: gym.spaces.Space,action_space: gym.spaces.Space,num_outputs: int,model_config: ModelConfigDict,name: str):
        TorchModelV2.__init__(self, obs_space,action_space,num_outputs,model_config,name)
        nn.Module.__init__(self)

        input_channel_size=19
        filters = 24
        res_blocks = 1
        se_channels = 0
        policy_conv_size = 73
        policy_output_size = 4672
        self.num_outputs = 4672
        self.name = name
        self.obs_space = obs_space
        self.action_space = action_space
        self.model_config = model_config

        self.input_channel_size = input_channel_size
        self.filters = filters
        self.res_blocks = res_blocks
        self.se_channels = se_channels
        self.policy_conv_size = policy_conv_size
        self.policy_output_size = policy_output_size
        self.pre_conv = nn.Conv2d(self.input_channel_size, self.filters, 3,padding = "same")
        self.conv1 = nn.Conv2d(self.filters, self.filters, 3,padding = "same")
        self.conv2 = nn.Conv2d(self.filters, self.filters, 3,padding = "same")
        self.pool = nn.AvgPool2d(8)
        self.se1 = nn.Linear(self.filters , self.se_channels)
        self.se2 = nn.Linear(self.se_channels,self.filters*2)
        self.fc_head = nn.Linear(self.filters*64,128)
        self.value_head = nn.Linear(128, 1)
        self.policy_conv1 = nn.Conv2d(self.filters, self.policy_conv_size, 3,padding = "same")
        self.policy_fc = nn.Linear(self.policy_conv_size*64, self.policy_output_size)
        self._value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.pre_conv(x)
        residual = x
        for i in range(self.res_blocks):
            x = self.conv1(x)
            x = self.conv2(x)
            if self.se_channels > 0:
                x = self.pool(x)
                x = torch.flatten(x, 1)
                x = F.relu(self.se1(x))
                x = self.se2(x)
                w,b = torch.tensor_split(x, 2,dim = -1)
                print(w.size(),b.size(),residual.size())
                residual = torch.reshape(residual, (-1,self.filters,64))
                x = torch.mul(w,residual) + b
            x += residual
            residual = x
            x = torch.relu(x)
        value = torch.flatten(x, 1)
        value = torch.relu(self.fc_head(value))
        value = torch.tanh(self.value_head(value))
        policy = self.policy_conv1(x)
        policy = torch.flatten(policy, 1)
        policy = self.policy_fc(policy)
        self._value = value.squeeze(1)
        
        return policy,state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self._value

    

