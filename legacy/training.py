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
from cloudops import *
from torch.utils.data.dataloader import default_collate

def val_value_model(agent,val_loader,optimizer,criterion,bce_criterion):
    agent.value_model.eval()
    running_loss = 0.0
    running_value_loss = 0.0
    running_policy_loss = 0.0
    j = 1
    for i, (inputs, labels,policy_labels) in enumerate(val_loader, 0):
        j += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs = inputs.to(CFG.DEVICE)
        labels = labels.to(CFG.DEVICE)
        policy_labels = policy_labels.to(CFG.DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        policy,value = agent.value_model(inputs)
        value_loss = criterion(value, labels)
        policy_loss = bce_criterion(policy,policy_labels)
        sum_loss = value_loss + policy_loss

        # print statistics
        running_loss += sum_loss.item()
        running_policy_loss += policy_loss.item()
        running_value_loss += value_loss.item()
    return running_value_loss/j,running_policy_loss/j

def validate_outcomes(agent,val_agents={},n_tries = 5,elo_save_treshold = 200):
    current_elo = 0
    for elo,val_agent in val_agents.items():
        current_elo = elo
        print(f"Validating at {elo} elo :")
        val_outcomes = experiments(agent,val_agent,n_tries,start_from_random=True,save_match_tensor=False)
        val_elo_diff = get_elo_diff_from_outcomes(val_outcomes)
        if val_elo_diff < elo_save_treshold:
            return val_agents
            break
        else:
            continue
    agent.elo_diff_from_random = current_elo + val_elo_diff
    agent.save_model()
    val_agent = agent.get_deepcopy()
    print("Saved model with an elo diff from random = "+str(agent.elo_diff_from_random))
    val_agents[agent.elo_diff_from_random] = val_agent
    return val_agents

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    data = torch.cat([item[0] for item in batch],dim = 0)
    value_target = torch.cat([item[1] for item in batch],dim = 0)
    policy_target = torch.cat([item[2] for item in batch],dim = 0)
    return [data,value_target,policy_target]

def train_value_model(agent,train_loader,val_loader=None,epochs = 1,progress_bar = True):
    if progress_bar:
        progress = tqdm(range(epochs), desc="")
    else:
        progress = range(range(epochs))
    criterion = CFG.criterion
    bce_criterion = CFG.bce_criterion
    optimizer = torch.optim.Adam(agent.value_model.parameters(),lr = CFG.lr,weight_decay = CFG.weight_decay)
    len_tl = len(train_loader)
    val_v_loss = 0
    val_p_loss = 0
    patience = 0
    max_patience = CFG.patience
    for epoch in progress:  # loop over the dataset multiple times
        agent.trained_epochs += 1
        agent.value_model.train()
        running_value_loss = 0.0
        running_policy_loss = 0.0
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
            policy,value = agent.value_model(inputs)
            
            policy_loss = CFG.weight_policy * bce_criterion(policy,policy_labels)
            value_loss = CFG.weight_value * criterion(value, labels)
            sum_loss = value_loss + policy_loss
            sum_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.value_model.parameters(), CFG.clip_norm)
            optimizer.step()
            
            # print statistics
            running_value_loss += value_loss.detach().item()
            running_policy_loss += policy_loss.detach().item()
            running_loss += sum_loss.detach().item()
            if progress_bar:
                progress.set_description(f"[{epoch+1}/{epochs}/{agent.trained_epochs}] [{i+1}/{len_tl}] value_loss: {running_value_loss/(i+1)} policy_loss: {running_policy_loss/(i+1)} val_value: {val_v_loss}  val_policy: {val_p_loss}")
        torch.cuda.empty_cache()
        if val_loader is not None:
            val_v_loss,val_p_loss = val_value_model(agent,val_loader,optimizer,criterion,bce_criterion)
            val_loss = val_v_loss + val_p_loss
            if val_loss < agent.best_val_loss:
                agent.best_val_loss = val_loss
                patience = 0
                if agent.elo_diff_from_random > 0:
                    agent.save_model()
            else:
                patience += 1
                if patience >= max_patience:
                    break
            torch.cuda.empty_cache()
            if progress_bar:
                progress.set_description(f"[{epoch+1}/{epochs}/{agent.trained_epochs}] [{i+1}/{len_tl}] value_loss: {running_value_loss/(i+1)} policy_loss: {running_policy_loss/(i+1)} val_value: {val_v_loss}  val_policy: {val_p_loss}")
        if (epoch+1) % CFG.every_x_epochs == 0:
            optimizer.param_groups[0]['lr'] *= CFG.mult_lr
    return None

def get_batch_datasets():
    train_dataset = BatchMemoryDataset(CFG.memory_batch[0:3])
    val_dataset = BatchMemoryDataset(CFG.memory_batch[3:6],CFG.val_last_index)
    return train_dataset,val_dataset

def get_tensors_from_files_datasets(dir_path,file_ending=".pt",train_idxs=None,val_idxs=None):
    val_dataset = CustomMatchDataset(dirpath = dir_path,idxs = val_idxs)
    train_dataset = CustomMatchDataset(dirpath = dir_path,idxs = train_idxs)
    return train_dataset,val_dataset

def get_data_loader(train_dataset,val_dataset=None,batch_size = 1,load_matches = False):
    if (CFG.save_tensor_to_disk and CFG.cloud_operations) or load_matches:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False,drop_last=False, num_workers=0,collate_fn=my_collate)
        if not val_dataset is None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,drop_last=False, num_workers=0,collate_fn=my_collate)
        else:
            val_loader = None
    else:
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,shuffle=True,drop_last=False, num_workers=0)
        if not val_dataset is None:
            val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size,drop_last=False, num_workers=0)
        else:
            val_loader = None
    return train_loader,val_loader

def get_master_datasets():
    file_list = [x for x in os.listdir(CFG.master_dataset_path) if x.endswith(".pt")]
    index_array = np.array([j for j in range(len(file_list))])
    kf = KFold(n_splits=100,shuffle=True,random_state=42)
    splits = list(kf.split(index_array))
    train_idxs,val_idxs = splits[0]
    train_dataset = MasterDataset(train_idxs)
    val_dataset = MasterDataset(val_idxs)
    return train_dataset,val_dataset

def train_master_games(agent,patience,epochs):
    prev_patience = CFG.patience
    CFG.patience = patience

    train_dataset,val_dataset = get_master_datasets()
    train_loader,val_loader = get_data_loader(train_dataset,val_dataset,load_matches=True,batch_size=8)
    train_value_model(agent,train_loader,val_loader,epochs = CFG.epochs)
    CFG.patience = prev_patience
    return None


def self_play(agent,base_agent=None,val_agent=None,play_batch_size = 4,n_episodes = 100,n_accumulate = 10):
    update_base_agent = False
    if CFG.load_best_model:
        register_model(CFG.model_dir_path,file_extension = ".pth",cleanup = False)
        f ,elo_diff = get_best_model(CFG.model_dir_path,file_extension = ".pth")
        # f = os.path.join(CFG.model_dir_path,"1041.0-194-10-best_model.pth")
        # elo_diff = 1041
        print("Loaded model with elo_diff: ",elo_diff)
        state_dict = torch.load(f,map_location = "cpu")
        agent.value_model.to(CFG.DEVICE)
        agent.value_model.load_state_dict(state_dict)
        val_agent = agent.get_deepcopy()
        val_agent.elo_diff_from_random = int(elo_diff)
        agent.elo_diff_from_random = int(elo_diff)
        val_elo = int(elo_diff)
    elif val_agent is None:
        val_elo = 0
        val_agent = RandomAgent()
    if base_agent is None:
        update_base_agent = True
        base_agent = agent.get_deepcopy()
    else:
        CFG.random_flip_chance = 0.5
    val_agents = {val_elo:val_agent}
    if CFG.start_master_train:
        train_master_games(agent,0,1)
    for episode in range(n_episodes):
        if (episode+1) % n_accumulate == 0:
            try:
                output.clear()
            except:
                pass
            delete_dataset_overflow(0,CFG.dataset_dir_path,file_ending=".pt")
            if update_base_agent:
                base_agent = agent.get_deepcopy()
        agent.value_model.train()
        CFG.batch_full = False
        agent.training = True
        train_outcomes = experiments(agent,base_agent,play_batch_size,start_from_random=True,random_start_depth = CFG.RANDOM_START)
        agent.training = False
        train_elo_diff = get_elo_diff_from_outcomes(train_outcomes)
        if not update_base_agent:
            a = 0
        if CFG.save_tensor_to_disk:
            file_list = [x for x in os.listdir(CFG.dataset_dir_path) if x.endswith(".pt")]
            index_array = np.array([j for j in range(len(file_list))])
            if (episode+1) % n_accumulate == 0 or episode == 0:
                kf = KFold(n_splits=10)
                splits = list(kf.split(index_array))
                train_idxs,val_idxs = splits[0]
            else:
                train_idxs = [index_array[x] for x in range(len(index_array)) if x not in val_idxs]
            train_dataset,val_dataset = get_tensors_from_files_datasets(CFG.dataset_dir_path,file_ending=".pt",train_idxs=train_idxs,val_idxs=val_idxs)
        else:
            train_dataset,val_dataset = get_batch_datasets()
        train_loader,val_loader = get_data_loader(train_dataset,val_dataset=val_dataset,batch_size = 1)
        train_value_model(agent,train_loader,val_loader,epochs = CFG.sampling_ratio)
        agent.value_model.eval()
        val_agents = validate_outcomes(agent,val_agents=val_agents)
        if CFG.cloud_operations:
            get_agent_pool_df(model_save_path = CFG.model_dir_path,file_extension = ".pth")
            delete_dataset_overflow(25,CFG.dataset_dir_path,file_ending=".pt")
    return None