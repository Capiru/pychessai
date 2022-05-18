import chess as ch
import numpy as np
import pandas as pd
import os


openings_dir = "./openings/"
openings_list = os.listdir(openings_dir)
openings_columns = ["starting_fen","name"]
for opening in openings_list:
    with open(openings_dir+opening) as f:
        print(f.read())

def build_openings_df(openings_dir = "./openings/",columns=["starting_fen","name"],save_df = True):
    openings_list = os.listdir(openings_dir)
    df = pd.DataFrame(columns=columns)
    for opening in openings_list:
        with open(openings_dir+opening) as f:
            df = df.append({columns[0]:f.read(),columns[1]:opening.replace(".fen","")},ignore_index=True)
    df = df.set_index(keys="starting_fen")
    if save_df:
        df.to_csv(openings_dir+"df_openings.csv")
    return df
build_openings_df()