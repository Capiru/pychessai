import os

def dir_size(dir_path):
    ### Returns the size of a directory in Gb
    return sum(os.path.getsize(f) for f in os.listdir(dir_path) if os.path.isfile(f))/1e9

def get_agent_pool_df(model_save_path,file_extension = ".pth"):
    file_list = [x for x in os.listdir(model_save_path) if file_extension in x]
    df = pd.DataFrame(columns=["val_loss","elo_diff","epochs"])
    i = 0
    for f in file_list:
        attributes = f.replace(file_extension,"").split("-")
        df.loc[len(df.index)] = attributes
    df.to_csv(model_save_path+"/"+"models.csv")