import pickle
import os
from trafficgen.utils.typedef import *
from scenarionet import read_dataset_summary, read_scenario
from trafficgen.utils.data_process.agent_process import WaymoAgent


path = '/home/ubuntu/DATA2/nuplan/boston/boston_0/sd_nuplan_v1.1_002dfa3f53805e1f.pkl' #'/home/ubuntu/DATA1/waymo/0_0.pkl'
path_root = '/home/ubuntu/DATA2/nuplan/boston/'
scenario_root = '/home/ubuntu/DATA2/nuplan/boston/boston_0/sd_nuplan_v1.1_'

from trafficgen.utils.data_process.agent_process import WaymoAgent
from IPython.display import Image as IImage
import pygame
import numpy as np
from PIL import Image

def make_GIF(frames, name="demo.gif"):
    print("Generate gif...")
    imgs = [frame for frame in frames]
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=50, loop=0)


with open(path, 'rb') as f:
    data = pickle.load(f)

tracks = data['tracks']


def valid_track(track, mask):
    mask = mask.astype(bool)
    return track[mask]


# ['position', 'heading', 'velocity', 'valid', 'length', 'width', 'height']
# 'track_length', 'nuplan_type', 'type', 'object_id', 'nuplan_id'

def extract_agents(tracks):
    type_vehicle = {"VEHICLE": 0, "TRAFFIC_CONE": 1, "PEDESTRIAN": 2, "CYCLIST": 3, "TRAFFIC_BARRIER": 4, "EGO": 5}

    agents_scenario = []

    for k, v in tracks.items():
    
        ts, _ = v['state']['position'].shape
        feature = np.ones((ts, 8))
        feature[:, :2] = v['state']['position'][:, :2]
        feature[:, 2:4] = v['state']['velocity']
        feature[:, 4] = v['state']['heading']
        feature[:, 5] = v['state']['length'][:, 0]
        feature[:, 6] = v['state']['width'][:, 0]
        feature[:, 7] = type_vehicle[v['type']]

        agent = WaymoAgent(feature)
        agents_scenario.append(agent)
    
    return agents_scenario

map_features = data["map_features"]

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from metadrive.type import MetaDriveType


def draw_map(map_features, show=False):
    figure(figsize=(8, 6), dpi=500)
    for key, value in map_features.items():
        if MetaDriveType.is_lane(value.get("type", None)):
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1)
        elif value.get("type", None) == "road_edge":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1, c=(0, 0, 0))
        # elif value.get("type", None) == "road_line":
        #     plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=1.0, c=(0.8,0.8,0.8))
        # break
    plt.savefig("demo_map.jpg")
    if show:
        plt.show()


# draw_map(map_features)
        

# dataset_summary, scenario_ids, mapping = read_dataset_summary(dataset_path=path_root)

def extract_poly(message):
    x = [i[0] for i in message]
    y = [i[1] for i in message]
    z = [-1 for i in message]
    coord = np.stack((x, y, z), axis=1)

    return coord


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype='int64')
    for k in range(len(fb)):
        c = dict()
        c['index'] = [fb[k].lane_start_index, fb[k].lane_end_index]
        c['type'] = RoadLineType(fb[k].boundary_type)
        c['id'] = fb[k].boundary_feature_id
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = fb[k].feature_id
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['boundaries'] = extract_boundaries(fb[k].boundaries)
        nb['id'] = fb[k].feature_id
        nbs.append(nb)
    return nbs

# dict_keys(['type', 'polyline', 'entry_lanes', 'exit_lanes', 'left_neighbor', 'right_neighbor', 'polygon'])
def extract_center(f):

    line_type2int = {"LANE_SURFACE_STREET":0, "LANE_SURFACE_UNSTRUCTURE": 1}

    center = {}

    poly = extract_poly(f['polyline'])
    poly = [np.insert(x, 2, line_type2int[f['type']]) for x in poly]

    # center['interpolating'] = f.interpolating

    center['entry'] = [x for x in f['entry_lanes']]

    center['exit'] = [x for x in f['exit_lanes']]

    # center['left_boundaries'] = extract_boundaries(f.left_boundaries)

    # center['right_boundaries'] = extract_boundaries(f.right_boundaries)

    # center['left_neighbor'] = extract_neighbors(f['left_neighbors'])

    # center['right_neighbor'] = extract_neighbors(f['right_neighbors'])

    return poly, center

def extract_map(f):
    maps = []
    center_infos = {}
    # nearbys = dict()
    for k, v in f.items():
        id = k
        if MetaDriveType.is_lane(v.get("type", None)):
            line, center_info = extract_center(v)
            center_infos[id] = center_info

        # elif f[i].HasField('road_line'):
        #     line = extract_line(f[i])

        # elif f[i].HasField('road_edge'):
        #     line = extract_edge(f[i])

        # elif f[i].HasField('stop_sign'):
        #     line = extract_stop(f[i])

        # elif f[i].HasField('crosswalk'):
        #     line = extract_crosswalk(f[i])

        # elif f[i].HasField('speed_bump'):
        #     line = extract_bump(f[i])
        # else:
        #     continue

        # line = [np.insert(x, 3, id) for x in line]
        # maps = maps + line

    return np.array(maps), center_infos

def extract_dynamic(dynamics):
    pass

# dict_keys(['id', 'version', 'length', 'metadata', 'tracks', 'dynamic_map_states', 'map_features'])
def parse_data(data):
    scene = dict()
    scene['id'] = data['id']
    scene['all_agent'] = extract_agents(data['tracks'])
    # scene['traffic_light'] = extract_dynamic(data['dynamic_map_states'])
    global SAMPLE_NUM
    SAMPLE_NUM = 10
    scene['lane'], scene['center_info'] = extract_map(data['map_features'])

    SAMPLE_NUM = 10e9
    scene['unsampled_lane'], _ = extract_map(data['map_features'])


# print(data['map_features'])

parse_data(data)
