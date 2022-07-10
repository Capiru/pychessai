import os
import pandas as pd
import shutil

def dir_size(dir_path):
    ### Returns the size of a directory in Gb
    return sum(os.path.getsize(f) for f in os.listdir(dir_path) if os.path.isfile(f))/1e9

def get_agent_pool_df(model_save_path,file_extension = ".pth"):
    file_list = [x for x in os.listdir(model_save_path) if file_extension in x and not x.endswith("-best_model.pth")]
    df = pd.DataFrame(columns=["val_loss","elo_diff","filters","res_blocks"])
    i = 0
    for f in file_list:
        attributes = f.replace(file_extension,"").split("-")
        df.loc[len(df.index)] = attributes
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
    shutil.copy(os.path.join(model_save_path,df.iloc[0]["val_loss"]+"-"+df.iloc[0]["elo_diff"]+"-"+df.iloc[0]["filters"]+df.iloc[0]["res_blocks"]+file_extension),os.path.join(model_save_path,f"{elo_diff}-best_model.pth"))
    if cleanup or dir_size(CFG.model_dir_path) > CFG.model_dir_size_limit:
        ### Delete all models except the best one
        for i in range(len(df.index)):
            os.remove(os.path.join(model_save_path,df.iloc[i]["val_loss"]+"-"+df.iloc[i]["elo_diff"]+"-"+df.iloc[i]["filters"]+df.iloc[i]["res_blocks"]+file_extension))