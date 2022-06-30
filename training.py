import torch
from config import CFG
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
# from google.colab import output
# from google.colab import drive
import shutil
from sklearn.model_selection import KFold
from match import *
from agents.RandomAgent import RandomAgent
from datasets import *

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
        running_loss += sum_loss.item()
    return running_loss/j

def validate_outcomes(agent,val_agents={},n_tries = 10,elo_save_treshold = 200):
    current_elo = 0
    for elo,val_agent in val_agents.items():
        print(f"Validating at {elo} elo :")
        current_elo = elo
        val_outcomes = experiments(agent,val_agent,n_tries,start_from_random=False,save_match_tensor=False)
        val_elo_diff = get_elo_diff_from_outcomes(val_outcomes)
        if val_elo_diff < elo_save_treshold:
            return val_agents
            break
        else:
            continue
    agent.elo_diff_from_random = current_elo + val_elo_diff
    agent.save_model(save_drive=CFG.cloud_operations,dir_path=CFG.model_dir_path)
    val_agent = agent.get_deepcopy()
    print("Saved model with an elo diff from random = "+str(agent.elo_diff_from_random))
    val_agents[agent.elo_diff_from_random] = val_agent
    return val_agents

def train_value_model(agent,train_dataset,val_dataset=None,progress_bar = True):
    if progress_bar:
        progress = tqdm(range(CFG.epochs), desc="")
    else:
        progress = range(range(CFG.epochs))
    criterion = CFG.criterion
    bce_criterion = CFG.bce_criterion
    optimizer = torch.optim.Adam(agent.value_model.parameters(),lr = CFG.lr,weight_decay = CFG.weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,shuffle=True,drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(CFG.batch_size/2),drop_last=False, num_workers=0)
    len_tl = len(train_loader)
    val_loss = 0
    patience = 0
    max_patience = CFG.patience
    for epoch in progress:  # loop over the dataset multiple times
        agent.trained_epochs += 1
        agent.value_model.train()
        running_loss = 0.0
        for i, (inputs, labels,policy_labels) in enumerate(train_loader, 0):
            sum_loss = 0
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
            running_loss += sum_loss.detach().item()
            if progress_bar:
                progress.set_description(f"[{epoch+1}/{CFG.epochs}/{agent.trained_epochs}] [{i+1}/{len_tl}] loss: {running_loss/(i+1)} val_loss: {val_loss}")
        val_loss = val_value_model(agent,val_loader,optimizer,criterion,bce_criterion)
        if (epoch+1) % CFG.every_x_epochs == 0:
            optimizer.param_groups[0]['lr'] *= CFG.mult_lr
        if val_loss < agent.best_val_loss and agent.elo_diff_from_random > 0:
            agent.best_val_loss = val_loss
            patience = 0
            agent.save_model()
        else:
            patience += 1
            if patience >= max_patience:
                break
    return None

def self_play(agent,base_agent=None,val_agent=None,play_batch_size = 4,n_episodes = 100,n_accumulate = 10):
    update_base_agent = False
    if val_agent is None:
        val_agent = RandomAgent()
    if base_agent is None:
        update_base_agent = True
        base_agent = agent.get_deepcopy()
    val_agents = {0:val_agent}
    for i in range(n_episodes):
        if (i+1) % n_accumulate == 0:
          output.clear()
          if update_base_agent:
              base_agent = agent.get_deepcopy()
          file_list = os.listdir("./")
          for item in file_list:
              if item.endswith(".pt"):
                  os.remove(os.path.join("./", item))
        agent.value_model.train()
        CFG.batch_full = False
        agent.training = True
        train_outcomes = experiments(agent,base_agent,play_batch_size,start_from_random=True,random_start_depth = 0)
        agent.training = False
        train_elo_diff = get_elo_diff_from_outcomes(train_outcomes)
        if not update_base_agent:
            a = 0
        if not CFG.save_batch_to_device:
            file_list = [x for x in os.listdir("./") if x.endswith(".pt")]
            index_array = np.array([j for j in range(len(file_list))])
            if (i+1) % n_accumulate == 0 or i == 0:
                kf = KFold(n_splits=10)
                splits = list(kf.split(index_array))
                train_idxs,val_idxs = splits[0]
                val_dataset = CustomMatchDataset(dirpath = "./",idxs = val_idxs)
            else:
                train_idxs = [index_array[x] for x in range(len(index_array)) if x not in val_idxs]
            train_dataset = CustomMatchDataset(dirpath = "./",idxs = train_idxs)
        else:
            train_dataset = BatchMemoryDataset(CFG.memory_batch[0:3])
            val_dataset = BatchMemoryDataset(CFG.memory_batch[3:6],CFG.val_last_index)
            
        train_value_model(agent,train_dataset,val_dataset)
        agent.value_model.eval()
        val_agents = validate_outcomes(agent,val_agents=val_agents)
        if CFG.cloud_operations:
            get_agent_pool_df(model_save_path = "/content/drive/MyDrive/projects/chessai",file_extension = ".pth")
    return None