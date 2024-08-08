import torch
import numpy as np
from trafficgen.utils.data_process.agent_process import WaymoAgent
from trafficgen.utils.visual_init import draw
from torch.utils.data.dataloader import default_collate
import dgl
import pickle
# try:
#   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# except:
#   tokenizer = None

def vis_init(data, agent_num=None):
  center = data["center"]
  rest = data["rest"]
  bound = data["bound"]
  agent = data["agent"]
  agents = [WaymoAgent(agent[i:i+1]) for i in range(agent.shape[0])]

  if agent_num:
    agents = agents[:agent_num]

  return draw(center, agents, other=rest, edge=bound, save_np=True, showup=False)

def get_degree_accel(traj, init_speed=0):
  last_speed = init_speed
  last_deg = 0
  
  accels = []
  degrees = []
  speeds = []

  step = traj.shape[0]
  for i in range(step-1):
    shift = traj[i+1] - traj[i]
    degree = np.rad2deg(np.arctan2(shift[1], shift[0]))
    degrees.append(degree-last_deg)
    last_deg = degree

    speed = np.linalg.norm(shift)
    accels.append(speed-last_speed)
    last_speed = speed
    speeds.append(speed)
  
  return degrees, accels, speeds

import pickle

def fc_collate_fn(batch):
  solid_path = "/home/ubuntu/xiajunkai/lctgen/lctgen/cheat.pkl"
  
  result_batch = {}
  
  extra_list = ['agent_feature', 'agent_type', 'pred_num', 'label', 'auxiliary_label', \
                        'label_mask', "other_label_mask", "map_fea", "scene_id", "obejct_id_lis", "other_label"]

  for key in batch[0].keys():
    if 'other' in key or 'center_info' in key:
        result_batch[key] = [item[key] for item in batch]
    else:
        
        if key in extra_list:
            continue
        
    result_batch[key] = default_collate([item[key] for item in batch])

  '''   
  try:
    hetero_graph = aggregate(batch)
    result_batch["hdgt_input"] = hetero_graph

    # with open(solid_path, "wb") as f:
    #     pickle.dump(result_batch, f)
    # print("dumped")
  except Exception as e:
    raise e
    with open(solid_path, "rb") as f:
        result_batch = pickle.load(f)
    print("loaded")
  '''
  return result_batch

map_size_lis = {1.0:30, 2.0:10, 3.0:20, -1.0:0, 0:0}


def euclid_np(label, pred):
    return np.sqrt((label[...,0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)

uv_dict = {}
## Sparse adj mat of fully connected graph of neighborhood size
def return_uv(neighborhood_size):
    global uv_dict
    if neighborhood_size in uv_dict:
        return uv_dict[neighborhood_size]
    else:
        v = torch.LongTensor([[_]*(neighborhood_size-1) for _ in range(neighborhood_size)]).view(-1)
        u = torch.LongTensor([list(range(0, _)) +list(range(_+1,neighborhood_size)) for _ in range(neighborhood_size)]).view(-1)
        uv_dict[neighborhood_size] = (u, v)
        return (u, v)

def generate_heterogeneous_graph(agent_fea, map_fea, agent_map_size_lis):
    max_in_edge_per_type = 32 ## For saving GPU memory
    uv_dic = {}
    uv_dic[("agent", "self", "agent")] = [list(range(agent_fea.shape[0])), list(range(agent_fea.shape[0]))] ## Self-loop
    num_of_agent = agent_fea.shape[0]
    ## Agent Adj
    uv_dic[("agent", "other", "agent")] = [[], []]
    for agent_index_i in range(num_of_agent):
        final_dist_between_agent = euclid_np(agent_fea[agent_index_i, -1, :][np.newaxis, :2], agent_fea[:, -1, :2])
        nearby_agent_index  = np.where(final_dist_between_agent < np.maximum(agent_map_size_lis[agent_index_i][np.newaxis], agent_map_size_lis))[0]
        nearby_agent_index = np.delete(nearby_agent_index, obj=np.where(nearby_agent_index == agent_index_i))
        if len(nearby_agent_index) > max_in_edge_per_type:
            final_dist_between_agent_sorted_nearby_index = np.argsort(final_dist_between_agent[nearby_agent_index])
            nearby_agent_index = nearby_agent_index[final_dist_between_agent_sorted_nearby_index][:max_in_edge_per_type]
        nearby_agent_index = nearby_agent_index.tolist()
        if len(nearby_agent_index) > 0:
            uv_dic[("agent", "other", "agent")][0] += [agent_index_i]*(len(nearby_agent_index))
            uv_dic[("agent", "other", "agent")][1] += nearby_agent_index

                   
    polygon_index_cnt = 0
    graphindex2polygonindex = {}
    uv_dic[("polygon", "g2a", "agent")] = [[], []]
    ## Agent_Polygon Adj
    if len(map_fea[1]) > 0:
        dist_between_agent_polygon = np.stack([(euclid_np(agent_fea[:, -1, :][:, np.newaxis, :], _[1][np.newaxis, :, :]).min(1)) for _ in map_fea[1]], axis=-1)
        all_agent_nearby_polygon_index_lis = dist_between_agent_polygon < agent_map_size_lis[:, np.newaxis]
        for agent_index_i in range(num_of_agent):
            nearby_polygon_index_lis = np.where(all_agent_nearby_polygon_index_lis[agent_index_i, :])[0]
            if len(nearby_polygon_index_lis) > max_in_edge_per_type:
                current_dist_between_agent_polygon = dist_between_agent_polygon[agent_index_i, :]
                nearby_polygon_index_lis_sorted = np.argsort(current_dist_between_agent_polygon[nearby_polygon_index_lis])
                nearby_polygon_index_lis = nearby_polygon_index_lis[nearby_polygon_index_lis_sorted][:max_in_edge_per_type]
            nearby_polygon_index_lis = nearby_polygon_index_lis.tolist()
            for now_cnt, nearby_polygon_index in enumerate(nearby_polygon_index_lis):
                uv_dic[("polygon", "g2a", "agent")][0].append(polygon_index_cnt)
                uv_dic[("polygon", "g2a", "agent")][1].append(agent_index_i)
                graphindex2polygonindex[polygon_index_cnt] = nearby_polygon_index
                polygon_index_cnt += 1

    laneindex2graphindex = {}
    graphindex_cnt = 0
    uv_dic[("lane", "l2a", "agent")] = [[], []]
    uv_dic[("agent", "a2l", "lane")] = [[], []]
    ## Agent-Map Adj
    if len(map_fea[0]) > 0:
        all_polyline_coor = np.array([_["xyz"] for _ in map_fea[0]])
        final_dist_between_agent_lane = euclid_np(agent_fea[:, -1, :2][:, np.newaxis, np.newaxis, :], all_polyline_coor[np.newaxis, :, :, :]).min(2)
        all_agent_nearby_lane_index_lis =  final_dist_between_agent_lane <  agent_map_size_lis[:, np.newaxis]
        for agent_index_i in range(num_of_agent):
            nearby_road_index_lis = np.where(all_agent_nearby_lane_index_lis[agent_index_i, :])[0]#.tolist()
            if len(nearby_road_index_lis) > max_in_edge_per_type:
                current_dist_between_agent_lane = final_dist_between_agent_lane[agent_index_i]
                nearby_road_index_lis_sorted = np.argsort(current_dist_between_agent_lane[nearby_road_index_lis])
                nearby_road_index_lis = nearby_road_index_lis[nearby_road_index_lis_sorted][:max_in_edge_per_type]
            nearby_road_index_lis = nearby_road_index_lis.tolist()
            for now_cnt, nearby_road_index in enumerate(nearby_road_index_lis):
                if nearby_road_index not in laneindex2graphindex:
                    laneindex2graphindex[nearby_road_index] = graphindex_cnt
                    graphindex_cnt += 1
                uv_dic[("agent", "a2l", "lane")][0].append(agent_index_i)
                uv_dic[("lane", "l2a", "agent")][1].append(agent_index_i)
                uv_dic[("lane", "l2a", "agent")][0].append(laneindex2graphindex[nearby_road_index])
                uv_dic[("agent", "a2l", "lane")][1].append(laneindex2graphindex[nearby_road_index])

    lane2lane_boundary_dic = {}
    ## Map-Map Adj
    for etype in ["prev", "follow"]:
        uv_dic[("lane", etype, "lane")] = [[], []]
        lane2lane_boundary_dic[("lane", etype, "lane")] = []
    if len(map_fea[0]) > 0:
        all_in_graph_lane = list(laneindex2graphindex.keys())
        for in_graph_lane in all_in_graph_lane:
            info_dic = map_fea[0][in_graph_lane]
            for etype in ["prev", "follow"]:
                neighbors = [_ for _ in info_dic[etype] if _[0] in laneindex2graphindex]
                lane2lane_boundary_dic[("lane", etype, "lane")] += [_[1] for _ in neighbors]
                neighbors = [_[0] for _ in neighbors]
                uv_dic[("lane", etype, "lane")][0] += [laneindex2graphindex[in_graph_lane]] * len(neighbors)
                uv_dic[("lane", etype, "lane")][1] += [laneindex2graphindex[_] for _ in neighbors]
    
    output_dic = {}
    for _ in uv_dic:
        uv_dic[_] = (torch.LongTensor(uv_dic[_][0]), torch.LongTensor(uv_dic[_][1]))

    output_dic["uv_dic"] = uv_dic
    output_dic["graphindex2polylineindex"] = {v: k for k, v in laneindex2graphindex.items()}
    output_dic["graphindex2polygonindex"] = graphindex2polygonindex
    output_dic["boundary_type_dic"] = {k:torch.LongTensor(v) for k, v in lane2lane_boundary_dic.items()}
    return output_dic

def rotate(data, cos_theta, sin_theta):
    data[..., 0], data[..., 1] = data[..., 0]*cos_theta - data[..., 1]*sin_theta, data[..., 1]*cos_theta + data[..., 0]*sin_theta
    return data

def normal_agent_feature(feature, ref_coor, ref_psi,  cos_theta, sin_theta):
    feature[..., :3] -= ref_coor[:, np.newaxis, :]
    feature[..., 0], feature[..., 1] = feature[..., 0]*cos_theta - feature[..., 1]*sin_theta, feature[..., 1]*cos_theta + feature[..., 0]*sin_theta
    feature[..., 3], feature[..., 4] = feature[..., 3]*cos_theta - feature[..., 4]*sin_theta, feature[..., 4]*cos_theta + feature[..., 3]*sin_theta
    feature[..., 5] -= ref_psi
    cos_psi = np.cos(feature[..., 5])
    sin_psi = np.sin(feature[..., 5])
    feature = np.concatenate([feature[..., :5], cos_psi[...,np.newaxis], sin_psi[...,np.newaxis], feature[..., 6:]], axis=-1)
    return feature

def normal_polygon_feature(all_polygon_coor, all_polygon_type, ref_coor, cos_theta, sin_theta):
    now_polygon_coor = all_polygon_coor - ref_coor
    rotate(now_polygon_coor, cos_theta, sin_theta)
    return now_polygon_coor,  all_polygon_type

def normal_lane_feature(now_polyline_coor, now_polyline_type, now_polyline_speed_limit, now_polyline_stop, now_polyline_signal, polyline_index, ref_coor, cos_theta, sin_theta):
    output_polyline_coor = now_polyline_coor[polyline_index] - ref_coor[:, np.newaxis, :]
    rotate(output_polyline_coor, cos_theta, sin_theta)
    output_stop_fea = {i:np.array(now_polyline_stop[_][0]) for i, _ in enumerate(polyline_index) if len(now_polyline_stop[_]) != 0}
    output_signal_fea = {i:np.array(now_polyline_signal[_][0]) for i, _ in enumerate(polyline_index) if len(now_polyline_signal[_]) != 0}
    output_stop_index, output_stop_fea = list(output_stop_fea.keys()), list(output_stop_fea.values())

    if len(output_stop_fea) != 0:
        output_stop_fea = np.stack(output_stop_fea, axis=0)
        output_stop_fea -= ref_coor[output_stop_index]
        if type(cos_theta) == np.float64:
            rotate(output_stop_fea, cos_theta, sin_theta)
        else:
            rotate(output_stop_fea, cos_theta[output_stop_index].flatten(), sin_theta[output_stop_index].flatten())

    output_signal_index, output_signal_fea = list(output_signal_fea.keys()), list(output_signal_fea.values())
    if len(output_signal_fea) != 0:
        output_signal_fea = np.stack(output_signal_fea, axis=0)
        output_signal_fea[..., :3] -= ref_coor[output_signal_index]
        if type(cos_theta) == np.float64:
            rotate(output_signal_fea, cos_theta, sin_theta)
        else:
            rotate(output_signal_fea, cos_theta[output_signal_index].flatten(), sin_theta[output_signal_index].flatten())
    return output_polyline_coor, now_polyline_type[polyline_index], now_polyline_speed_limit[polyline_index], output_stop_fea, output_stop_index, output_signal_fea, output_signal_index

def return_rel_e_feature(src_ref_coor, dst_ref_coor, src_ref_psi, dst_ref_psi):
    rel_coor = src_ref_coor - dst_ref_coor
    if rel_coor.ndim == 0 or rel_coor.ndim == 1:
        rel_coor = np.atleast_1d(rel_coor)[np.newaxis, :]
    rel_coor = rotate(rel_coor, np.cos(-dst_ref_psi),  np.sin(-dst_ref_psi))
    rel_psi = np.atleast_1d(src_ref_psi - dst_ref_psi)[:, np.newaxis]
    rel_sin_theta = np.sin(rel_psi)
    rel_cos_theta = np.cos(rel_psi)
    return np.concatenate([rel_coor, rel_sin_theta, rel_cos_theta], axis=-1)


map_size_lis = {1.0:30, 2.0:10, 3.0:20, -1.0:0, 0.0:0}
@torch.no_grad()

def type_to_int(tp):
    type_2_int = {"VEHICLE":1.0, "PEDESTRIAN":2.0, "CYCLIST":3.0, "TRAFFIC_BARRIER":-1.0, "TRAFFIC_CONE":-1.0}
    return float(tp)

import matplotlib.pyplot as plt
def aggregate(batch, is_train = True):
    agent_drop = 0.0
    agent_feature_lis = [item["agent_feature"] for item in batch]
    agent_type_lis = [item["agent_type"] for item in batch]
    #agent_map_size_lis = [np.vectorize(setting_dic["agenttype2mapsize"].get)(_) for _ in agent_type_lis]
    pred_num_lis = np.array([item["pred_num"] for item in batch])

    # label_lis = [item["label"] for item in batch] # [:, :49, :]
    
    label_lis = [item['traj'].swapaxes(0,1)[item['agent_mask']] for item in batch]

    # for i, item in enumerate(batch):
    #     points = item["label"][0, :49, :].tolist()

    #     # folder_path = "./nuplan_vis/"
    #     # x_coords = [point[0] for point in points]
    #     # y_coords = [point[1] for point in points]

    #     # plt.plot(x_coords, y_coords, marker='o') 
    #     # path = folder_path + str(i) + ".jpg"
    #     # print(path)
    #     # plt.savefig(path)
    #     print(f"{item['agent_feature'].shape=}")
    #     print(f"{item['pred_num']=}")
    #     print(f"{item['traj'].swapaxes(0,1)[item['agent_mask']].shape=}")
    #     print(f"{item['label'].shape=}")

    auxiliary_label_lis =  [item["auxiliary_label"][:, :50, :] for item in batch] # [:, :49, :]
    label_mask_lis = [item["label_mask"][:, :50] for item in batch] # [:, :49]
    other_label_lis = [item["other_label"] for item in batch]
    other_label_mask_lis = [item["other_label_mask"] for item in batch]
    map_fea_lis = [item["map_fea"] for item in batch]
    case_id_lis = [item["scene_id"] for item in batch]
    object_id_lis = [item["obejct_id_lis"] for item in batch]

    if agent_drop > 0 and is_train:
        for i in range(len(agent_feature_lis)):
            keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            while keep_index[:pred_num_lis[i]].sum() == 0:
                keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            origin_pred_num = pred_num_lis[i]
            original_agent_num = agent_feature_lis[i].shape[0]
            target_keep_index = keep_index[:origin_pred_num]
            agent_feature_lis[i] = agent_feature_lis[i][keep_index]
            
            agent_type_lis[i] = agent_type_lis[i][keep_index]
            pred_num_lis[i] = int(target_keep_index.sum())

            label_lis[i] = label_lis[i][target_keep_index]
            auxiliary_label_lis[i] = auxiliary_label_lis[i][target_keep_index]
            label_mask_lis[i] = label_mask_lis[i][target_keep_index]
            if origin_pred_num != original_agent_num:
                other_label_lis[i] = other_label_lis[i][keep_index[origin_pred_num:]]
                other_label_mask_lis[i] = other_label_mask_lis[i][keep_index[origin_pred_num:]]
    
    neighbor_size = np.array([int(agent_feature_lis[i].shape[0]) for i in range(len(agent_feature_lis))])

    out_lane_n_stop_sign_fea_lis = []
    out_lane_n_stop_sign_index_lis = []
    out_lane_n_signal_fea_lis = []
    out_lane_n_signal_index_lis = []

    out_normal_lis = []
    out_graph_lis = []
    out_label_lis = []
    out_label_mask_lis = []
    out_auxiliary_label_lis = []
    out_auxiliary_label_future_lis = []
    out_other_label_lis = []
    out_other_label_mask_lis = []
    lane_n_cnt = 0

    for i in range(len(agent_feature_lis)):
        all_agent_obs_final_v = np.sqrt(agent_feature_lis[i][:, -1, 3]**2+agent_feature_lis[i][:, -1, 4]**2)
        type_temp = agent_type_lis[i]
        agent_type = [type_to_int(tp) for tp in type_temp]
        all_agent_map_size = np.vectorize(map_size_lis.__getitem__)(agent_type)
        all_agent_map_size = all_agent_obs_final_v * 8.0 + all_agent_map_size

        graph_dic = generate_heterogeneous_graph(agent_feature_lis[i], map_fea_lis[i], all_agent_map_size)
        g = dgl.heterograph(data_dict=graph_dic["uv_dic"])
        g.edata['boundary_type'] = graph_dic["boundary_type_dic"]

        polylinelaneindex = list(graph_dic["graphindex2polylineindex"].values())
        polygonlaneindex = list(graph_dic["graphindex2polygonindex"].values())
        now_agent_feature = agent_feature_lis[i]
        now_agent_type = np.array([type_to_int(x) for x in  agent_type_lis[i]], dtype=np.float64)
       
        

        ### Type 0 edge a2a self-loop
        type0_u, type0_v = g.edges(etype="self")
        now_t0_v_feature = now_agent_feature[type0_v, :, :]
        now_t0_e_feature = now_agent_feature[type0_u].copy()
        if len(type0_v) == 1:
            now_t0_v_feature = now_t0_v_feature[np.newaxis, :, :]
            now_t0_e_feature = now_t0_e_feature[np.newaxis, :, :]
        now_t0_e_feature = return_rel_e_feature(now_t0_e_feature[:, -1, :3], now_t0_v_feature[:, -1, :3], now_t0_e_feature[:, -1, 5], now_t0_v_feature[:, -1, 5])
        g.edata['a_e_fea'] = {("agent", "self", "agent"):torch.as_tensor(now_t0_e_feature.astype(np.float32))}
        g.edata['a_e_type'] = {("agent", "self", "agent"):torch.as_tensor((now_agent_type[type0_u].ravel()-1).astype(np.int32)).long()}
        
        ### Type 0 edge a2a other agent
        type1_u, type1_v = g.edges(etype="other")
        if len(type1_v) > 0:
            now_t1_v_feature = now_agent_feature[type1_v, :, :]
            now_t1_e_feature = now_agent_feature[type1_u].copy()
            if len(type1_v) == 1:
                now_t1_v_feature = now_t1_v_feature[np.newaxis, :, :]
                now_t1_e_feature = now_t1_e_feature[np.newaxis, :, :]
            now_t1_e_feature = return_rel_e_feature(now_t1_e_feature[:, -1, :3], now_t1_v_feature[:, -1, :3], now_t1_e_feature[:, -1, 5], now_t1_v_feature[:, -1, 5])
            g.edata['a_e_fea'] = {("agent", "other", "agent"):torch.as_tensor(now_t1_e_feature.astype(np.float32))}
            g.edata['a_e_type'] = {("agent", "other", "agent"):torch.as_tensor((now_agent_type[type1_u].ravel()-1).astype(np.int32)).long()}
        else:
            g.edata['a_e_fea'] = {("agent", "other", "agent"):torch.zeros((0, 5))}
            g.edata['a_e_type'] = {("agent", "other", "agent"):torch.zeros((0, )).long()}

        ### Type 2 Edge: Agent -> Lane  a2l
        if len(polylinelaneindex) > 0:
            now_polyline_info = [map_fea_lis[i][0][_] for _ in polylinelaneindex]
            now_polyline_coor = np.stack([_["xyz"] for _ in now_polyline_info], axis=0)
            now_polyline_yaw = np.array([_["yaw"] for _ in now_polyline_info])
            now_polyline_type = np.array([_["type"] for _ in now_polyline_info])
            now_polyline_speed_limit = np.array([_["speed_limit"] for _ in now_polyline_info])
            now_polyline_stop = [_["stop"] for _ in now_polyline_info]
            now_polyline_signal = [_["signal"] for _ in now_polyline_info]
            now_polyline_mean_coor = now_polyline_coor[:, 2, :]
            type2_u = g.edges(etype="a2l")[0]#[0][cumu_edge_type_cnt_lis[2]:cumu_edge_type_cnt_lis[3]]
            type2_v = g.edges(etype="a2l")[1]#[1][cumu_edge_type_cnt_lis[2]:cumu_edge_type_cnt_lis[3]] - now_agent_feature.shape[0] - len(polygonlaneindex)
            if len(type2_v) > 0:
                now_t2_e_feature = now_agent_feature[type2_u].copy()
                if len(now_t2_e_feature.shape) == 2:
                    now_t2_e_feature = now_t2_e_feature[np.newaxis, :, :]
                now_t2_e_feature = return_rel_e_feature(now_t2_e_feature[:, -1, :3], now_polyline_mean_coor[type2_v], now_t2_e_feature[:, -1, 5], now_polyline_yaw[type2_v])
                g.edata['a_e_fea'] = {("agent", "a2l", "lane"):torch.as_tensor(now_t2_e_feature.astype(np.float32))}
                g.edata['a_e_type'] = {("agent", "a2l", "lane"):torch.as_tensor((now_agent_type[type2_u].ravel()-1).astype(np.int32)).long()}
               

        ### Type 3 Edge: Polygon -> Agent  g2a
        type3_u = g.edges(etype="g2a")[0]#[cumu_edge_type_cnt_lis[3]:cumu_edge_type_cnt_lis[4]]  - now_agent_feature.shape[0]
        type3_v = g.edges(etype="g2a")[1]#[cumu_edge_type_cnt_lis[3]:cumu_edge_type_cnt_lis[4]]
        if len(type3_v) > 0:
            now_polygon_type = np.array([map_fea_lis[i][1][_][0] for _ in polygonlaneindex])
            now_polygon_coor = np.stack([map_fea_lis[i][1][_][1] for _ in polygonlaneindex], axis=0)
            now_t3_v_feature = now_agent_feature[type3_v]
            if len(now_t3_v_feature.shape) == 2:
                now_t3_v_feature = now_t3_v_feature[np.newaxis, :, :]
            ref_coor = now_t3_v_feature[:, -1, :3][:, np.newaxis, :]
            ref_psi = now_t3_v_feature[:, -1, 5][:, np.newaxis].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_t3_e_coor_feature, now_t3_e_type_feature = normal_polygon_feature(now_polygon_coor, now_polygon_type, ref_coor, cos_theta, sin_theta)
            g.edata['g2a_e_fea'] = {("polygon", "g2a", "agent"):torch.as_tensor(now_t3_e_coor_feature.astype(np.float32))}
            g.edata['g2a_e_type'] = {("polygon", "g2a", "agent"):torch.as_tensor(now_t3_e_type_feature.ravel().astype(np.int32)).long()}

        ### Type 4 Edge: Lane -> Agent            
        if len(polylinelaneindex) > 0:
            type4_u = g.edges(etype="l2a")[0]
            type4_v = g.edges(etype="l2a")[1]
            if len(type4_v) > 0:
                now_t4_v_feature = now_agent_feature[type4_v]
                if len(now_t4_v_feature.shape) == 2:
                    now_t4_v_feature = now_t4_v_feature[np.newaxis, :, :]
                now_t4_e_feature = return_rel_e_feature(now_polyline_mean_coor[type4_u], now_t4_v_feature[:, -1, :3], now_polyline_yaw[type4_u], now_t4_v_feature[:, -1, 5])
                g.edata['l_e_fea'] = {("lane", "l2a", "agent"):torch.as_tensor(now_t4_e_feature.astype(np.float32))}

        ### Type 5 Edge: Lane -> Lane
        if len(polylinelaneindex) > 0:
            for etype in ["prev", "follow"]:
                type5_u = g.edges(etype=etype)[0]
                type5_v = g.edges(etype=etype)[1]
                if len(type5_v) > 0:
                    now_t5_e_feature = return_rel_e_feature(now_polyline_mean_coor[type5_u], now_polyline_mean_coor[type5_v], now_polyline_yaw[type5_u], now_polyline_yaw[type5_v])
                    g.edata['l_e_fea'] = {("lane", etype, "lane"):torch.as_tensor(now_t5_e_feature.astype(np.float32))}

        now_pred_num = pred_num_lis[i]
        selected_pred_indices = list(range(0, now_pred_num))
        non_pred_indices = list(range(now_pred_num, now_agent_feature.shape[0]))
        
        ## Label + Full Agent Feature
        now_full_agent_n_feature = now_agent_feature[selected_pred_indices].copy()
        ref_coor = now_full_agent_n_feature[:, -1,:3].copy()
        now_label = label_lis[i][selected_pred_indices].copy()
        now_auxiliary_label = auxiliary_label_lis[i][selected_pred_indices].copy()
        now_label = now_label - ref_coor[:, np.newaxis, :2]
        ref_psi = now_full_agent_n_feature[:, -1, 5][:, np.newaxis].copy()
        normal_val = np.concatenate([ref_coor[..., :2], ref_psi], axis=-1)
        out_normal_lis.append(normal_val)
        
        sin_theta = np.sin(-ref_psi)
        cos_theta = np.cos(-ref_psi)
        rotate(now_label, cos_theta, sin_theta)
        rotate(now_auxiliary_label, cos_theta, sin_theta)
        now_auxiliary_label[..., 2] = now_auxiliary_label[..., 2] - ref_psi
        
        now_full_agent_n_feature = normal_agent_feature(now_full_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta)
        now_auxiliary_label_future = now_auxiliary_label.copy()
        now_auxiliary_label = np.stack([now_full_agent_n_feature[..., 3],  now_full_agent_n_feature[..., 4], now_agent_feature[selected_pred_indices, :, 5]-ref_psi, now_full_agent_n_feature[..., -1]], axis=-1)


        now_all_agent_n_feature = now_full_agent_n_feature
        if now_pred_num < now_agent_feature.shape[0]:
            now_other_agent_n_feature = now_agent_feature[non_pred_indices].copy()
            ref_coor = now_other_agent_n_feature[:, -1, :3]
            ref_psi = now_other_agent_n_feature[:, -1, 5][:, np.newaxis].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_other_agent_n_feature = normal_agent_feature(now_other_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta)
            now_all_agent_n_feature = np.concatenate([now_all_agent_n_feature, now_other_agent_n_feature], axis=0)
        g.ndata["a_n_fea"] = {"agent":torch.as_tensor(now_all_agent_n_feature.astype(np.float32))}
        g.ndata["a_n_type"] = {"agent":torch.as_tensor((now_agent_type-1).astype(np.int32)).long()}
        
        line_type2int = {"LANE_SURFACE_STREET": 0, "LANE_SURFACE_UNSTRUCTURE": 1, 'ROAD_LINE_BROKEN_SINGLE_WHITE': 2, 'ROAD_LINE_SOLID_SINGLE_WHITE': 3}

        ## Lane Node Feature
        if len(polylinelaneindex) > 0:
            ref_coor = now_polyline_mean_coor
            ref_psi = now_polyline_yaw[:, np.newaxis].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_lane_n_coor_feature, now_lane_n_type_feature, now_lane_n_speed_limit_feature, now_lane_n_stop_feature, now_lane_n_stop_index, now_lane_n_signal_feature, now_lane_n_signal_index = normal_lane_feature(now_polyline_coor, now_polyline_type, now_polyline_speed_limit, now_polyline_stop, now_polyline_signal, list(range(len(now_polyline_coor))), ref_coor, cos_theta, sin_theta)                
            now_lane_n_type_feature = np.array([tp for tp in now_lane_n_type_feature], dtype = np.int32)
            g.ndata["l_n_coor_fea"] = {"lane":torch.as_tensor(now_lane_n_coor_feature.astype(np.float32))}
            g.ndata["l_n_type_fea"] = {"lane":torch.as_tensor(now_lane_n_type_feature.astype(np.int32)).long()}

        ## Polyline Feature
        if len(polylinelaneindex) > 0:
            if len(now_lane_n_stop_index) != 0:
                out_lane_n_stop_sign_fea_lis.append(now_lane_n_stop_feature)
                out_lane_n_stop_sign_index_lis.append(np.array(now_lane_n_stop_index) + lane_n_cnt)
            if len(now_lane_n_signal_index) != 0:
                out_lane_n_signal_fea_lis.append(now_lane_n_signal_feature)
                out_lane_n_signal_index_lis.append(np.array(now_lane_n_signal_index)+lane_n_cnt)
            lane_n_cnt += now_lane_n_coor_feature.shape[0]

        out_graph_lis.append(g)
        out_label_lis.append(now_label)
        out_label_mask_lis.append(label_mask_lis[i][selected_pred_indices])
        out_auxiliary_label_lis.append(now_auxiliary_label)
        out_auxiliary_label_future_lis.append(now_auxiliary_label_future)

    output_dic = {}
    #0-x, 1-y, 2-vx, 3-vy, 4-cos_psi, 5-sin_psi, 6-length, 7-width, 8-type, 9-mask
    output_dic["cuda_tensor_lis"] = ["graph_lis"]
    output_dic["cuda_tensor_lis"] += ["label_lis", "label_mask_lis", "auxiliary_label_lis", "auxiliary_label_future_lis"]
    if len(out_lane_n_stop_sign_fea_lis) > 0:
        output_dic["cuda_tensor_lis"] += ["lane_n_stop_sign_fea_lis", "lane_n_stop_sign_index_lis"]
        out_lane_n_stop_sign_index_lis = np.concatenate(out_lane_n_stop_sign_index_lis, axis=0)
        output_dic["lane_n_stop_sign_fea_lis"] = torch.as_tensor(np.concatenate(out_lane_n_stop_sign_fea_lis, axis=0).astype(np.float32))
        output_dic["lane_n_stop_sign_index_lis"] =  torch.as_tensor(out_lane_n_stop_sign_index_lis.astype(np.int32)).long()

    if len(out_lane_n_signal_fea_lis) > 0:
        output_dic["cuda_tensor_lis"] += ["lane_n_signal_fea_lis", "lane_n_signal_index_lis"]
        out_lane_n_signal_index_lis = np.concatenate(out_lane_n_signal_index_lis, axis=0)
        output_dic["lane_n_signal_fea_lis"] = torch.as_tensor(np.concatenate(out_lane_n_signal_fea_lis, axis=0).astype(np.float32))
        output_dic["lane_n_signal_index_lis"] =  torch.as_tensor(out_lane_n_signal_index_lis.astype(np.int32)).long()
    output_dic["label_lis"] = torch.as_tensor(np.concatenate(out_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_lis"] = torch.as_tensor(np.concatenate(out_auxiliary_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_future_lis"] = torch.as_tensor(np.concatenate(out_auxiliary_label_future_lis, axis=0).astype(np.float32))

    output_dic["label_mask_lis"] = torch.as_tensor(np.concatenate(out_label_mask_lis, axis=0).astype(np.float32))

    output_g = dgl.batch(out_graph_lis)
    a_e_type_dict = {}
    for out_etype in ["self", "a2l", "other"]:
        a_e_type_dict[out_etype] = []
        for agent_tpye_index in range(3):
            a_e_type_dict[out_etype].append(torch.where(output_g.edges[out_etype].data["a_e_type"]==agent_tpye_index)[0])
    a_n_type_lis = [torch.where(output_g.nodes["agent"].data["a_n_type"]==_)[0] for _ in range(3)]
    output_dic["a_e_type_dict"] = a_e_type_dict
    output_dic["a_n_type_lis"] = a_n_type_lis
    output_dic["graph_lis"] = output_g
    output_dic["neighbor_size_lis"] = neighbor_size
    output_dic["pred_num_lis"] = pred_num_lis
    output_dic["case_id_lis"] = case_id_lis
    output_dic["object_id_lis"] = object_id_lis
    output_dic["normal_lis"] = np.concatenate(out_normal_lis, axis=0)
    if "fname" in batch[0]:
        all_filename = [item["fname"] for item in batch]
        output_dic["fname"] = []
        for _ in range(len(all_filename)):
            output_dic["fname"] += [all_filename[_]]*pred_num_lis[_]
    del batch
    return output_dic
