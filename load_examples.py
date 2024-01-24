import os
data_path = "/home/ubuntu/DATA1/waymo"
save_path = "/home/ubuntu/xiajunkai/lctgen/data/list/map.txt"
import pickle
import numpy as np

path = "/home/ubuntu/DATA1/waymo/demo_map_vec.npy"
data = np.load(path, allow_pickle=True)

file_name = data.item()['ids']
with open(save_path, 'w') as f:
    for file in file_name:
        lst_f = file.split("_")
        file_path = str(lst_f[0]) + "_" + str(lst_f[1]) + ".pkl"
        f.write(file_path)
        f.write("\n")
    f.close()

'''
with open(save_path, 'w') as f:
    for data in os.listdir(data_path):
        if ".npy" in str(data):
            continue
        f.write(data)
        f.write("\n")
    f.close()
'''
