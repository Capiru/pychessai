import torch
from config import CFG

def val_value_model(agent,val_loader,optimizer,criterion):
    agent.value_model.eval()
    running_loss = 0.0
    j = 1
    for i, (inputs, labels) in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = inputs.to(CFG.DEVICE)
        labels = labels.to(CFG.DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        value,policy = agent.value_model(inputs)
        loss = criterion(value, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        j = i+1
    return running_loss/j