import torch
from config import CFG

def val_value_model(agent,val_loader,optimizer,criterion,bce_criterion):
    agent.value_model.eval()
    running_loss = 0.0
    j = 1
    for i, (inputs, labels,policy_labels) in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs = inputs.to(CFG.DEVICE)
        labels = labels.to(CFG.DEVICE)
        policy_labels = policy_labels.to(CFG.DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        value,policy = agent.value_model(inputs)
        value_loss = criterion(value, labels)
        policy_loss = bce_criterion(policy,policy_labels)
        sum_loss = value_loss + policy_loss
        sum_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss/j