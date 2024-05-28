import pickle
import os
from trafficgen.utils.typedef import *
from scenarionet import read_dataset_summary, read_scenario
from trafficgen.utils.data_process.agent_process import WaymoAgent


path = '/home/ubuntu/DATA2/nuplan/boston/_0/sd_nuplan_v1.1_011968a1408c5ae4.pkl' #'/home/ubuntu/DATA1/waymo/0_0.pkl'
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
        feature = np.ones((ts, 9))
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

BATCH_SIZE = 190


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
    z = [0 for i in message]
    coord = np.stack((x, y, z), axis=1)

    return coord


# def extract_boundaries(fb):
#     b = []
#     # b = np.zeros([len(fb), 4], dtype='int64')
#     for k in range(len(fb)):
#         c = dict()
#         c['index'] = [fb[k].lane_start_index, fb[k].lane_end_index]
#         c['type'] = RoadLineType(fb[k].boundary_type)
#         c['id'] = fb[k].boundary_feature_id
#         b.append(c)

#     return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = k
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

def down_sampling(line, type=0):
    # if is center lane
    point_num = len(line)

    ret = []

    if point_num < SAMPLE_NUM or type == 1:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0, point_num, SAMPLE_NUM):
            ret.append(line[i])

    return ret

line_type2int = {"LANE_SURFACE_STREET": 0, "LANE_SURFACE_UNSTRUCTURE": 1, 'ROAD_LINE_BROKEN_SINGLE_WHITE': 2, 'ROAD_LINE_SOLID_SINGLE_WHITE': 3}

# dict_keys(['type', 'polyline', 'entry_lanes', 'exit_lanes', 'left_neighbor', 'right_neighbor', 'polygon'])
def extract_center(f):
    # plt.scatter([x[0] for x in f["polyline"]], [y[1] for y in f["polyline"]], s=0.1)
    center = {}

    poly = down_sampling(extract_poly(f['polyline'])[:, :2])
    poly = [np.insert(x, 2, line_type2int[f['type']]) for x in poly]

    center['interpolating'] = [] #f.interpolating

    center['entry'] = [x for x in f['entry_lanes']]

    center['exit'] = [x for x in f['exit_lanes']]

    if "left_boundaries" in f.keys():
        center['left_boundaries'] = [] # extract_boundaries(f["left_boundaries"])
    else:
        center['left_boundaries'] = []

    if "right_boundaries" in f.keys():
        center['right_boundaries'] = [] # extract_boundaries(f["right_boundaries"])
    else:
        center['right_boundaries'] = []
    
    if "left_neighbor" in f.keys():
        center['left_neighbor'] = [] # extract_neighbors(f['left_neighbor'])
    else:
        center['left_neighbor'] = []

    if "right_neighbor" in f.keys():
        center['right_neighbor'] = [] # extract_neighbors(f['right_neighbor'])
    else:
        center['right_neighbor'] = []

    

    return poly, center

def extract_boundaries(f):

    poly = down_sampling(extract_poly(f['polyline'])[:, :2])
    type = line_type2int[f['type']] + 5
    poly = [np.insert(x, 2, type) for x in poly]

    return poly

def extract_crosswalk(f):
    poly = down_sampling(extract_poly(f['polygon'])[:, :2], 1)
    poly = [np.insert(x, 2, 18) for x in poly]
    return poly

def extract_map(f):
    figure(figsize=(8, 6), dpi=500)
    maps = []
    center_infos = {}
    # nearbys = dict()
    for k, v in f.items():
        id = k
        if MetaDriveType.is_lane(v.get("type", None)):            
            line, center_info = extract_center(v)
            center_infos[id] = center_info

        elif MetaDriveType.is_road_line(v.get("type", None)): 
            line = extract_boundaries(v)

        elif MetaDriveType.is_crosswalk(v.get("type", None)): 
            line = extract_crosswalk(v)
        
        else:
            continue
        
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

        try:
            line = [np.insert(x, 3, int(id)) for x in line]
            
        except:
            line = [np.insert(x, 3, -1) for x in line] # try
       

        # print(len(line))
        maps.append(line)
        
    


    maps = np.vstack(maps)
    # plt.savefig("lane_map.jpg")
    return maps, center_infos

def extract_dynamic(f):
    dynamics = []
    for k,v in f.items():
        # states = f[i * time_sample].lane_states
        traf_list = np.zeros(6)
        states = v['state']
        traf_list[0] = int(v['lane'])
        traf_list[1:4] = np.array([[v['start_point'][0], v['stop_point'][1], 0]])
        if states['object_state'] == 'TRAFFIC_LIGHT_RED':
            state_ = 1  # stop
        elif states['object_state'] == 'TRAFFIC_LIGHT_YELLOW':
            state_ = 2  # caution
        elif states['object_state'] == 'TRAFFIC_LIGHT_GREEN':
            state_ = 3  # go
        else:
            state_ = 0  # unknown
        
        traf_list[4] = state_
        traf_list[5] = 1 if v['state'] else 0

        dynamics.append(traf_list)
    return dynamics

def nearest_point(point, line):
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return np.argmin(dist)

def extract_width(map, polyline, boundary):
    l_width = np.zeros(polyline.shape[0])
    for b in boundary:
        idx = map[:, -1] == b['id']
        b_polyline = map[idx][:, :2]

        start_p = polyline[b['index'][0]]
        start_index = nearest_point(start_p, b_polyline)
        seg_len = b['index'][1] - b['index'][0]
        end_index = min(start_index + seg_len, b_polyline.shape[0] - 1)
        leng = min(end_index - start_index, b['index'][1] - b['index'][0]) + 1
        self_range = range(b['index'][0], b['index'][0] + leng)
        bound_range = range(start_index, start_index + leng)
        centerLane = polyline[self_range]
        bound = b_polyline[bound_range]
        dist = np.square(centerLane - bound)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        l_width[self_range] = dist
    return l_width

def compute_width(scene):
    lane = scene['unsampled_lane']
    lane = lane.reshape((-1, 1))
    lane_id = np.unique(lane[..., -1]).astype(int)
    center_infos = scene['center_info']
   
    for id in lane_id:
        if not id in center_infos.keys():
            continue
        id_set = lane[..., -1] == id
        points = lane[id_set]

        width = np.zeros((points.shape[0], 2))

        width[:, 0] = extract_width(lane, points[:, :2], center_infos[id]['left_boundaries'])
        width[:, 1] = extract_width(lane, points[:, :2], center_infos[id]['right_boundaries'])

        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]

        center_infos[id]['width'] = width
    return

# dict_keys(['id', 'version', 'length', 'metadata', 'tracks', 'dynamic_map_states', 'map_features'])
def parse_data(data):
    scene = dict()
    scene['id'] = data['id']
    scene['all_agent'] = extract_agents(data['tracks'])
    scene['traffic_light'] = extract_dynamic(data['dynamic_map_states'])
    global SAMPLE_NUM
    SAMPLE_NUM = 10
    scene['lane'], scene['center_info'] = extract_map(data['map_features'])
    SAMPLE_NUM = 10e9
    scene['unsampled_lane'], _ = extract_map(data['map_features'])

    # compute_width(scene)
    return scene


# scenario = parse_data(data)

# print(scenario['id'])
# # d683acddb18a9cdf
# print(len(scenario['all_agent']))
# # (190, N, 9)
# print(len(scenario['traffic_light'][0]))
# # 190 * 6
# print(scenario['lane'].shape)
# # (1740, 4)
# print(scenario['center_info'].keys())
# # dict_keys([105, 106, 116, 117,...])
# print(scenario['center_info']['48607'].keys())
# # dict_keys(['interpolating', 'entry', 'exit', 'left_boundaries', 'right_boundaries', 'left_neighbor', 'right_neighbor', 'width'])
# print(scenario['unsampled_lane'].shape)
# # (15638, 4)

# target_file = "test.pkl"
# with open(target_file, "wb") as t:
#     pickle.dump(scenario, t)


import numpy as np

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.component.scenario_block.scenario_block import ScenarioBlock
from metadrive.engine.asset_loader import AssetLoader
from metadrive.type import MetaDriveType
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.math import resample_polyline, get_polyline_length
from metadrive.utils.draw_top_down_map import draw_top_down_map

from metadrive.engine.engine_utils import initialize_engine
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.manager.scenario_data_manager import ScenarioDataManager

import cv2 as cv

default_config = ScenarioEnv.default_config()
# default_config["_render_mode"] = "onscreen"
default_config["use_render"] = False
default_config["debug"] = False
default_config["debug_static_world"] = True
default_config["data_directory"] = path_root
default_config["num_scenarios"] = 1
engine = initialize_engine(default_config)

# engine.data_manager = ScenarioDataManager()


m_data = data["map_features"]
map = ScenarioMap(map_index=0, map_data=m_data)

m = draw_top_down_map(map)
cv.imwrite("feature.jpg", m)
