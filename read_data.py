import pickle
from trafficgen.utils.typedef import *
path = './data/demo/waymo/0_4693.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
