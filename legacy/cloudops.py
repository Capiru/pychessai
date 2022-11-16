import os
import pandas as pd
import shutil
from config import CFG


def dir_size(dir_path):
    ### Returns the size of a directory in Gb
    return sum(os.path.getsize(f) for f in os.listdir(dir_path) if os.path.isfile(f))/1e9

def get_agent_pool_df(model_save_path,file_extension = ".pth",remove_best_model = True):
    file_list = [x for x in os.listdir(model_save_path) if file_extension in x]
    if remove_best_model:
        file_list = [x for x in file_list if "best_model" not in x]
        df = pd.DataFrame(columns=["val_loss","elo_diff","filters","res_blocks"])
        len_ = 4
    else:
        file_list = [x for x in file_list if "best_model" in x]
        df = pd.DataFrame(columns=["elo_diff","filters","res_blocks"])
        len_ = 3
    i = 0
    for f in file_list:
        attributes = f.replace(file_extension,"").split("-")
        df.loc[len(df.index)] = [float(x) for x in attributes[0:len_]]
    df.to_csv(model_save_path+"/"+"models.csv")
    return df

def get_agent_pool_df_from_csv(model_save_path):
    df = pd.read_csv(model_save_path+"/"+"models.csv")
    return df

def register_model(model_save_path,file_extension = ".pth",cleanup = False):
    df = get_agent_pool_df(model_save_path,file_extension)
    ### Sort by elo diff
    df = df.sort_values(by=["elo_diff"],ascending=False)
    ### Register model
    if len(df.index) > 0:
        elo_diff = str(int(df.iloc[0]["elo_diff"]))
        filters = str(int(df.iloc[0]["filters"]))
        res_blocks = str(int(df.iloc[0]["res_blocks"]))
        shutil.copy(os.path.join(model_save_path,str(df.iloc[0]["val_loss"])+"-"+elo_diff+"-"+filters+"-"+res_blocks+file_extension),os.path.join(model_save_path,f"{elo_diff}-{filters}-{res_blocks}-best_model.pth"))
        if cleanup or dir_size(CFG.model_dir_path) > CFG.model_dir_size_limit:
            ### Delete all models except the best one
            for i in range(len(df.index)):
                os.remove(os.path.join(model_save_path,df.iloc[i]["val_loss"]+"-"+df.iloc[i]["elo_diff"]+"-"+df.iloc[i]["filters"]+df.iloc[i]["res_blocks"]+file_extension))

### return best model based on elo diff
def get_best_model(model_save_path,file_extension = ".pth"):
    df = get_agent_pool_df(model_save_path,file_extension,remove_best_model=False)
    ### Sort by elo diff
    df = df.sort_values(by=["elo_diff"],ascending=False)
    ### Register model
    if len(df.index) > 0:
        elo_diff = str(int(df.iloc[0]["elo_diff"]))
        filters = str(int(df.iloc[0]["filters"]))
        res_blocks = str(int(df.iloc[0]["res_blocks"]))
        return os.path.join(model_save_path,f"{elo_diff}-{filters}-{res_blocks}-best_model.pth"),df.iloc[0]["elo_diff"]
    else:
        return None,None

### append logs in a file
def append_log(log_path,log_string):
    with open(log_path,"a") as f:
        f.write(log_string)
        f.write("\n")

### delete dataset overflow
def delete_dataset_overflow(size,dataset_path,file_ending = ".pt"):
    if dir_size(CFG.dataset_dir_path) > size:
        file_list = os.listdir(CFG.dataset_dir_path)
        for item in file_list:
            if item.endswith(".pt"):
                os.remove(os.path.join(CFG.dataset_dir_path, item))