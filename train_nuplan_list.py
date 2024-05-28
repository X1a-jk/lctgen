import os
base_path = "/home/ubuntu/DATA2/nuplan/boston/_0"

target_path = "/home/ubuntu/xiajunkai/lctgen/data/list/train_nuplan.txt"

files = os.listdir(base_path)

with open(target_path, "w") as f:
    for file in files:
        f.write(file)
        f.write("\n")

    f.close()