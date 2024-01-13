import os
data_path = "/home/ubuntu/dataset/waymo"
save_path = "/home/ubuntu/xiajunkai/lctgen/data/list/0.txt"

with open(save_path, 'w') as f:
    for data in os.listdir(data_path):
        if ".npy" in str(data):
            continue
        f.write(data)
        f.write("\n")
    f.close()
