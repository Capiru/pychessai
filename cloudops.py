import os

def dir_size(dir_path):
    ### Returns the size of a directory in Gb
    return sum(os.path.getsize(f) for f in os.listdir(dir_path) if os.path.isfile(f))/1e9

