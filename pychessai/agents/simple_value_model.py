import chess as ch
import torch
import torch.nn as nn
import torch.nn.functional as F
from match import match, get_fen_as_tensor
from move_choice import minimax_with_pruning
from typing import Callable

from pychessai.agents import TrainableAgent
from pychessai.move_choice.utils import legal_moves
from pychessai.move_choice.minimax import minimax
from pychessai.models import Model

class SimpleTrainableAgent(TrainableAgent):
    def __init__(
        self,
        depth: int,
        board: ch.Board,
        is_white: bool,
        search_function: Callable = minimax,
        policy_function: Callable = legal_moves,
        eval_function: Callable = lambda x : x,
        training: bool= True,
        model_class : Model = SimpleModel,
        model_parameters: dict = {},
    ):
        super().__init__(depth,board, is_white, search_function,policy_function,eval_function,training,model_parameters)
        self.setup_model()

    def choose_move(self, board):
        self.board = board
        score, move, positions = self.search_function(
            self.board,
            self.depth,
            self.is_white,
            eval_function=self.eval_function,
            policy_function=self.policy_function,
        )
        self.positions += positions
        self.eval = score
        return [move]
    
    def get_board_evaluation(self,board):
        return self.model.predict(board)
    
    def setup_model(self) -> None:
        self.model = Model(self.model_parameters)
    
    def create_match_data(self,board,is_white,match_status):
        if match_status == 1 and is_white:
            #winner
            out = 1
        elif match_status == 0:
            out = 0.5
        else:
            out = 0


    def create_dataset(filepath):
        return dataset

    def create_dataloader(dataset):
        #trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return dataloader

    def train_value_model(input_tensor,match_outcome):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.ADAM(self.value_model.parameters())
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.value_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training')
        return None
    
    def train_policy_model(input_tensor,move_output):
        return None

    def inference_policy_model(input_tensor):
        #also might need to reshape input tensor
        self.policy_model.eval()
        outputs = self.policy_model(input_tensor)
        return None

    def legal_inference(legal_moves,move_outputs):
        #first transform move_outputs to san move
        output_reshaped = torch.reshape(move_outputs,(8,8,12))
        dic_encoder = {"p":0,"P":1,"r":2,"R":3,"n":4,"N":5,"b":6,"B":7,"q":8,"Q":9,"k":10,"K":11}
        row_encoder = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H"}
        #i need to get the max, index
        #then compare if move is in legal moves
        for move in move_outputs:
            if list(legal_moves):
                return move

    def inference_value_model(input_tensor):
        #i might need to reshape this to pass to the model
        self.value_model.eval()
        outputs = self.value_model(input_tensor)
        return outputs


    class ValueModel(nn.Module):
        def __init__(self):
            super().__init__()
            input_channel_size = 12
            self.conv1 = nn.Conv2d(input_channel_size, 24, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(24, 32, 5)
            self.fc1 = nn.Linear(32 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(self.conv2(x))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def get_board_evaluation(self,board):
            input_tensor = get_fen_as_tensor(board)
            #might need to reshape for input
            return self.forward(input_tensor)

    class PolicyModel(nn.Module):
        #this will need to be an encoder decoder, output will need to be the same size as input
        def __init__(self):
            super().__init__()
            input_channel_size = 12
            self.conv1 = nn.Conv2d(input_channel_size, 24, 3)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(24, 32, 5)
            self.fc1 = nn.Linear(32 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 256)
            self.fc3 = nn.Linear(256, 8*8*12)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
