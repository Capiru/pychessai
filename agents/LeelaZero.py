import chess as ch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from match import match, get_board_as_tensor
from move_choice import minimax_with_pruning_and_policyeval
from search.search import MonteCarloSearchNode
import shutil
from config import CFG

class LeelaZero(nn.Module):
    def __init__(self,input_channel_size=19,filters = 48,res_blocks = 6,se_channels = 0,policy_conv_size = 73,policy_output_size = 4672):
        super().__init__()
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

    def forward(self, x):
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
        
        return value,policy
    
    def get_board_evaluation(self,board,is_player_white):
        input_tensor = get_board_as_tensor(board,is_player_white)
        #might need to reshape for input
        c,w,h = input_tensor.shape
        return self.forward(torch.reshape(input_tensor,[1,c,w,h]))

class LeelaZeroAgent(object):
    def __init__(self,n_simulations = 30,board = ch.Board(),is_white = True,batch_size = 4,epochs = 3,training = True,
                input_channel_size=19,filters = 48,res_blocks = 6,se_channels = 0,policy_conv_size = 73,policy_output_size = 4672):
        super().__init__()
        self.elo = 400
        self.n_simulations = n_simulations
        self.board = board
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_white = is_white
        self.training = training
        self.policy_output_size = policy_output_size
        self.positions = 0
        self.eval = 0
        self.best_val_loss = np.inf
        self.trained_epochs = 0
        self.elo_diff_from_random = 0
        self.input_channel_size,self.filters,self.res_blocks,self.se_channels,self.policy_conv_size = input_channel_size,filters,res_blocks,se_channels,policy_conv_size
        self.value_model = LeelaZero(input_channel_size=self.input_channel_size,filters=self.filters,res_blocks=self.res_blocks,se_channels=self.se_channels,
                                    policy_conv_size=self.policy_conv_size,policy_output_size=self.policy_output_size)
        self.parent_node = None

    def choose_move(self,board):
        self.board = board
        self.parent_node = MonteCarloSearchNode(self,None,not self.is_white,board)
        score,move = self.parent_node.search(n_simulations = self.n_simulations)
        # try:
        #     self.parent_node = self.parent_node.children[move]
        # except:
        #     self.parent_node = None
        self.positions += self.n_simulations
        try:
            self.eval = score.detach().cpu()
        except:
            self.eval = score
        return [move]
    
    def get_model_name(self):
        return str(round(self.best_val_loss,8))+"-"+str(self.elo_diff_from_random)+"-"+str(self.trained_epochs)+".pth"

    def save_model(self,save_drive = True,dir_path = "/content/drive/MyDrive/projects/chessai"):
        model_name = self.get_model_name()
        torch.save(self.value_model.state_dict(), model_name)
        if save_drive:
            shutil.copy(model_name,dir_path)

    def get_deepcopy(self):
        new_agent = LeelaZeroAgent(self.n_simulations,self.board,self.is_white,self.batch_size,self.epochs,self.training,
                self.input_channel_size,self.filters,self.res_blocks,self.se_channels,self.policy_conv_size)
        state_dict = self.value_model.state_dict()
        new_agent.value_model.load_state_dict(state_dict)
        new_agent.value_model.to(CFG.DEVICE)
        return new_agent

    def reset_game(self):
        self.parent_node = None
