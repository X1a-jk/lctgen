import pickle
from trafficgen.utils.typedef import *
path = '/home/ubuntu/DATA1/waymo/0_0.pkl'
print(path)
with open(path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
