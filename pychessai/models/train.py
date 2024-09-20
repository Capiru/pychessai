from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TrainingParameters:
    batch_size: int = 8
    shuffle_data: bool = False


class Trainer(ABC):
    def __init__(self, criterion, optimizer, training_parameters):
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_parameters = training_parameters

    # train ValueModel
    # The ValueModel is trying to predict if the game is going to be a win, draw or lose.
    # Options are:
    # - MSE Regress the reward
    # - Classify (logits probability) of 3 classes (win, draw, lose)

    def train_value_model(self, input_tensor, match_outcome):
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
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0
        print("Finished Training")
        return None


# train PolicyModel
# The PolicyModel is trying to predict which is the best next move for the Agent.
# Options are:
# - Classification (CrossEntropy) based on best move and predicted move
# - Update the probabilities based on what MCTS has found
