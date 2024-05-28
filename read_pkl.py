import os
import pickle
from trafficgen.utils.typedef import *
# file = '/home/ubuntu/DATA1/waymo/0_1.pkl'
for i in range(10):
    path = f'/home/ubuntu/DATA2/nuplan/boston/_{i}/'
    files = os.listdir(path)
    for f in files:
        file = path + f
        if 'dataset' in file:
            continue
        with open(file, 'rb') as f:
            data = pickle.load(f)
            if 'dynamic_map_states' not in data.keys():
                print(file)
                continue
            dynamics = data['dynamic_map_states']
            if len(dynamics) != 0:
                print(file)

print(data.keys())
# dict_keys(['id', 'all_agent', 'traffic_light', 'lane', 'center_info', 'unsampled_lane'])
'''
print(data['id'])
# d683acddb18a9cdf
print(data['all_agent'].shape)
# (190, N, 9)
print(type(data['traffic_light'][0]))
# 190 * 14
print(type(data['lane']))
# (1740, 4)
print(data['center_info'].keys())
# dict_keys([105, 106, 116, 117,...])
print(data['center_info'][227]['left_neighbor'])
# dict_keys(['interpolating', 'entry', 'exit', 'left_boundaries', 'right_boundaries', 'left_neighbor', 'right_neighbor', 'width'])
print(data['unsampled_lane'].shape)
# (15638, 4)
'''
