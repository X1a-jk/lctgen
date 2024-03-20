import pickle
import os
from trafficgen.utils.typedef import *
path = '/home/ubuntu/DATA2/nuplan/pittsburgh/processed_0/sd_nuplan_v1.1_0003725baa1d5693.pkl' #'/home/ubuntu/DATA1/waymo/0_0.pkl'
path_root = '/home/ubuntu/DATA2/nuplan/pittsburgh/'
print(path)
'''
with open(path, 'rb') as f:
    data = pickle.load(f)
'''

'''
for k, v in data.items():
    print(k)
    
    if type(v) is dict:
        continue
    print(v)
    print("---------------")
    
'''

# id, version, length, metadata, tracks, dynamic_map_states, map_features
'''
for k, v in data['tracks'].items():
    print(v['state']['position'])
    break
'''
# VEHICLE, TRAFFIC_CONE, PEDESTRIAN, CYCLIST, TRAFFIC_BARRIER
files = os.listdir(path_root)

type_scenario = {"VEHICLE": 0, "TRAFFIC_CONE": 0, "PEDESTRIAN": 0, "CYCLIST": 0, "TRAFFIC_BARRIER": 0}
type_agent = {"VEHICLE": 0, "TRAFFIC_CONE": 0, "PEDESTRIAN": 0, "CYCLIST": 0, "TRAFFIC_BARRIER": 0}

for file in files:
    if file.endswith(".pkl"):
        continue
    scenes_path = path_root+file
    for scene in os.listdir(scenes_path):
        scene_f = scenes_path + "/" + scene
        type_scene = []
        with open(scene_f, 'rb') as f:
            scenario = pickle.load(f)
            if not ('tracks' in scenario.keys()):
                continue
            for idx, veh in scenario['tracks'].items():
                veh_type = veh['metadata']['type']
                type_agent[veh_type] += 1
                if not (veh_type in type_scene):
                    type_scene.append(veh_type)
            f.close()
        for tp in type_scene:
            type_scenario[tp] += 1

normalize = True
if normalize:
    veh_scenario = type_scenario["VEHICLE"]
    for t, p in type_scenario.items():
        type_scenario[t] /= veh_scenario

    veh_agent = type_agent["VEHICLE"]
    for t, p in type_agent.items():
        type_agent[t] /= veh_agent

print(type_scenario)
print(type_agent)


