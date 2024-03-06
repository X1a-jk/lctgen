import torch
import numpy as np
from .interaction_type import Interaction


def kmeans_label(data, k, max_time = 10):
    init_pos_batch = []
    init_heading_batch = []
    init_vel_batch = []
    init_type_batch = []
    for j in range(data['gt_pos'][:, data['agent_mask'], :].shape[1]):
        # print("***************************************************")
        init_pos_batch.append(data['gt_pos'][:, data['agent_mask'], :][:, j, :][0])
        # print(data['gt_pos'][:, data['agent_mask'][0], :][:, j, :][0])
        init_heading_batch.append(data['future_heading'][:, data['agent_mask']][:, j][0])
        init_vel_batch.append(np.linalg.norm(data['future_vel'][:, data['agent_mask'], :][:, j, :][0]))
        init_type_batch.append(data['traj_type'][data['agent_mask'], :][j, :][0])

    init_pos_batch = torch.tensor(np.vstack(init_pos_batch))
    #print(init_pos_batch.shape)
    init_heading_batch = torch.tensor(np.vstack(init_heading_batch))
    #print(init_heading_batch.shape)
    init_vel_batch = torch.tensor(np.vstack(init_vel_batch))
    #print(init_vel_batch.shape)
    init_type_batch = torch.tensor(np.vstack(init_type_batch)).float()
    #print(init_type_batch.shape)

    n, m = init_pos_batch.shape
    ini = torch.randint(n, (k,))
    midpoint = init_pos_batch[ini]
    time = 0
    last_label = 0
    while(time < max_time):
        d = init_pos_batch.unsqueeze(0).repeat(k, 1, 1)
        mid_= midpoint.unsqueeze(1).repeat(1,n,1)
        dis = torch.sum((d - mid_)**2, 2) 
        label = dis.argmin(0)      #依据最近距离标记label
        if torch.sum(label != last_label)==0:  #label没有变化,跳出循环
            return label, init_pos_batch, init_heading_batch, init_vel_batch, init_type_batch  
        last_label = label
        for i in range(k):  #更新类别中心点，作为下轮迭代起始
            kpoint = init_pos_batch[label==i] 
            if i == 0:
                midpoint = kpoint.mean(0).unsqueeze(0)
            else:
                midpoint = torch.cat([midpoint, kpoint.mean(0).unsqueeze(0)], 0)
        time += 1
    #print(label)
    return label, init_pos_batch, init_heading_batch, init_vel_batch, init_type_batch

def collect_data(data):
    init_pos_batch = []
    init_heading_batch = []
    init_vel_batch = []
    init_type_batch = []
    for j in range(data['gt_pos'][:, data['agent_mask'], :].shape[1]):
        # print("***************************************************")
        init_pos_batch.append(data['gt_pos'][:, data['agent_mask'], :][:, j, :][0])
        # print(data['gt_pos'][:, data['agent_mask'][0], :][:, j, :][0])
        init_heading_batch.append(data['future_heading'][:, data['agent_mask']][:, j][0])
        init_vel_batch.append(np.linalg.norm(data['future_vel'][:, data['agent_mask'], :][:, j, :][0]))
        init_type_batch.append(data['traj_type'][data['agent_mask'], :][j, :][0])

    init_pos_batch = torch.tensor(np.vstack(init_pos_batch))
    #print(init_pos_batch.shape)
    init_heading_batch = torch.tensor(np.vstack(init_heading_batch))
    #print(init_heading_batch.shape)
    init_vel_batch = torch.tensor(np.vstack(init_vel_batch))
    #print(init_vel_batch.shape)
    init_type_batch = torch.tensor(np.vstack(init_type_batch)).float()
    #print(init_type_batch.shape)

    return init_pos_batch, init_heading_batch, init_vel_batch, init_type_batch

def attr_fuse(data, k, label, max_agents, init_pos, init_heading, init_vel, init_type, dimension):
    #num_veh in cluster, avg speed, avg_heading, cluster center[x, y], max_beh
    # 每个类给一个feature
    default = -1 * torch.ones((max_agents, dimension))
    for i in range(k):
        pos = init_pos[label==i]
        if pos.shape[0] < 1:
            continue
        vel = init_vel[label==i]
        heading = init_heading[label==i]
        tp = init_type[label==i]
        default[i, 0] = pos.shape[0] #num_vehicles
        default[i, 1] = vel.mean()
        # default[i, 2] = heading.mean()
        default[i, 2:4] = pos.mean()
        default[i, 4] = tp.mean()
        
    return default

def binary_attr_fuse(data, default, max_agents, init_pos, init_heading, init_vel, init_type, dimension):
    label_x = 0
    num_veh = init_pos.shape[0]
    for i in range(num_veh):
        pos_i = init_pos[i]
        head_i = init_heading[i] + torch.arctan(pos_i[1] / pos_i[0]) if pos_i[0] != 0.0 else init_heading[i] + torch.pi / 2.0
        vel_i = init_vel[i]
        type_i = init_type[i]
        for j in range(num_veh):
            if i == j:
                continue
            pos_j = init_pos[j]
            head_j = init_heading[j] + torch.arctan(pos_j[1] / pos_j[0]) if pos_j[0] != 0.0 else init_heading[j] + torch.pi / 2.0
            vel_j = init_vel[j]
            type_j = init_type[j]
            mean_pos = (pos_i + pos_j) / 2
            mean_head = (head_i + head_j) / 2
            mean_vel = (vel_i + vel_j) / 2
            sort_id = label_x #i*max_agents+j
            default[sort_id, 0:2] = mean_pos
            default[sort_id, 2] = mean_head
            default[sort_id, 3] = mean_vel
            default[sort_id, 4] = type_i
            default[sort_id, 5] = type_j
            label_x += 1
    default_mask = np.zeros((max_agents**2), dtype=bool)
    default_mask[0:num_veh**2] = 1
    return default_mask

def _get_all_traj(data, action_step, s_rate=None, sample_num=None):
    VALID_LIMIT = 100
    trajs = data['gt_pos']
    all_heading = data["future_heading"][:, data['agent_mask']]
    traj_each_agent = {}
    heading_each_agent = {}
    for aix in range(trajs[:, data['agent_mask'], :].shape[1]):
        pos_agent = trajs[:, data['agent_mask'], :][:, aix, :]
        heading_agent = all_heading[:, aix]
        valid_mask = (abs(pos_agent[:, 0])<VALID_LIMIT) * (abs(pos_agent[:, 1])<VALID_LIMIT)
        pos_agent = pos_agent[valid_mask]
        pos_step = pos_agent.shape[0]
        if s_rate == None:
          sample_rate = pos_step // (action_step+1)
        else:
          sample_rate = s_rate 
        if sample_num == None:
          sample_num = -1
        
        final_pos = pos_agent[-1]
        pos_agent = pos_agent[::sample_rate][:sample_num]
        pos_agent[-1] = final_pos
        
        traj_each_agent.update({aix: pos_agent})
        heading_agent = heading_agent[valid_mask]
        heading_agent = heading_agent[::sample_rate][:sample_num].reshape((-1,1))

        for i in range(heading_agent.shape[0]):
            if pos_agent[i, 1] == 0.0:
                continue
            elif pos_agent[i, 0] == 0.0:
                heading_agent[i, 0] += np.pi / 2
            else:
                heading_agent[i, 0] += np.arctan(pos_agent[i, 1] / pos_agent[i, 0])
        
        heading_each_agent.update({aix: heading_agent})

    return traj_each_agent, heading_each_agent

def _get_type_traj(data, action_step, s_rate=None, sample_num=None):
    VALID_LIMIT = 100
    trajs = data['gt_pos']
    all_type = data["traj_type"][data['agent_mask']]

    return all_type

def _get_neighbor_text(data, default, max_agents):
    # print(self.data['file'])
    SAMPLE_NUM = 5
    '''
    if len(self.data['agent_mask']) == 1:
        all_trajs = self.data['traj']
    else:
        all_trajs = self.data['traj'][:, self.data['agent_mask']]
    trajs = all_trajs#[:, self.sorted_idx]
    if 'all_agent_mask' not in self.data:
        traj_masks = np.ones_like(trajs[:, :, 0]) == True
    else:
        traj_masks = self.data['all_agent_mask'][:, self.data['agent_mask']][:, self.sorted_idx]
    '''
    action_step = 4
    action_dim = 1
    # future_angles = np.cumsum(self.data["future_heading"], axis=0)
    trajs = data['gt_pos']
    all_heading = data["future_heading"][:, data['agent_mask']]
    traj_each_agent = {}
    heading_each_agent = {}
    traj_each_agent, heading_each_agent = _get_all_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    default_mask = np.zeros((max_agents**2), dtype=bool)
    # print(heading_each_agent)
    if len(traj_each_agent) <= 1:
        default_mask[0] = 1
        return default_mask

    # default = -1 * torch.ones((max_agents, SAMPLE_NUM, 2))
    # default[0, :] = torch.zeros((1, SAMPLE_NUM, 2)) # both dis, pos
    # default[0, :, 0] = 0  # w/o dis
    # default[0, :, 1] = 0  # w/o pos
    num_veh = len(heading_each_agent)
    label_x = 0
    for i in range(num_veh):
        ego_heading = heading_each_agent[i]      
        # neighbor_trajs_tensor = -1 * torch.ones((max_agents, SAMPLE_NUM * 2))
        ego_traj = traj_each_agent[i]
    # neighbor_trajs_tensor[0, :] = torch.zeros((1, SAMPLE_NUM * 2)) # both dis, pos
    # neighbor_trajs_tensor[0, SAMPLE_NUM:] = torch.zeros((1, SAMPLE_NUM)) # w/o dis
    # neighbor_trajs_tensor[0, :SAMPLE_NUM] = torch.zeros((1, SAMPLE_NUM)) # w/o pos
        
        for aidx in range(len(traj_each_agent)):
            if i == aidx:
                label_x += 1
                continue
            traj_temp = traj_each_agent[aidx]
            lst_temp = []
            for time_step in range(traj_temp.shape[0]):
                ego_pos = ego_traj[time_step]
                current_pos = traj_temp[time_step]
                current_pos_rel = pos_rel(ego_heading[time_step], ego_pos, current_pos)
                # neighbor_trajs_tensor[aidx][time_step] = current_pos_rel
                lst_temp.append(current_pos_rel)
            ll = []
            for j in lst_temp:
                ll.append(j[0])
            for k in lst_temp:
                ll.append(k[1])
            ll.append(-1)
            default[label_x] = torch.tensor(ll)
            label_x += 1
        # neighbor_trajs_tensor[aidx] = torch.tensor(ll)
    # neighbor_trajs_tensor = neighbor_trajs_tensor.view((max_agents, -1))
    default_mask[0:num_veh**2] = 1
    return default_mask
    
def pos_rel(ego_heading, ego_pos, other_pos):
    angle_init = ego_heading[0]
    pos_rel = other_pos - ego_pos
    dis_rel = np.linalg.norm(pos_rel)
    degree_pos_rel = int(np.clip(dis_rel/2.5, a_min=0, a_max=8))
    if pos_rel[1] == 0:
        deg_other = 0
    elif pos_rel[0] == 0:
        deg_other = np.pi/2
    else:
        deg_other = np.arctan2(pos_rel[1], pos_rel[0])
    deg_rel = deg_other - ego_heading
    if deg_rel > np.pi:
        deg_rel -= 2 * np.pi
    elif deg_rel < -1 * np.pi:
        deg_rel += 2*np.pi
        
    if deg_rel < np.pi/6 and deg_rel > -1 * np.pi / 6:
        ang = 0
    elif deg_rel <= -1 * np.pi / 6 and deg_rel > -1 * np.pi / 2:
        ang = 1
    elif  deg_rel <= -1 * np.pi / 2 and deg_rel > -5 * np.pi/6:
        ang = 2
    elif deg_rel >= np.pi/6 and deg_rel < np.pi/2:
        ang = 5
    elif deg_rel >= np.pi/2 and deg_rel < 5*np.pi/6:
        ang = 4
    else:
        ang = 3
    # degree_pos_rel = -1 # w/o relstive distance
    # ang = -1 # w/o relative pos
    return [degree_pos_rel, ang]



def kmeans_fuse(data, k, max_time = 10, max_agents = 32, dimension = 5):
    default = -1 * torch.ones((max_agents, dimension))
    try:
        label, init_pos, init_heading, init_vel, init_type = kmeans_label(data, k, max_time)
        km_feat = attr_fuse(data, k, label, max_agents, init_pos, init_heading, init_vel, init_type, dimension)
        return km_feat
    except Exception as e:
        print(e)
        return default
    
def binary_fuse(data, max_agents = 32, dimension = 6):
    default = -1 * torch.ones((max_agents**2, dimension))

    try:
        init_pos, init_heading, init_vel, init_type = collect_data(data)
        binary_mask = binary_attr_fuse(data, default, max_agents, init_pos, init_heading, init_vel, init_type, dimension)
        return default, binary_mask
    except Exception as e:
        raise(e)
        return default
    
def star_fuse(data, max_agents = 32, dimension = 5*2+1):
    default = -1 * torch.ones((max_agents**2, dimension))
    try:
        star_mask = _get_neighbor_text(data, default, max_agents)
        return default, star_mask
    except Exception as e:
        raise(e)
        return default
    
def get_type_interactions(data, max_agents = 32):
    inter_type = _get_inter_type(data, max_agents)
    return inter_type

def type_traj(traj):
    stop_lim = 1.0
    lane_width = 4.0
    ang_lim = 15
    valid_traj, _ = traj.shape
    if valid_traj<=2:
        traj_type = -1
        return traj_type
    
    pos_init = traj[0]
    pos_final = traj[-1]

    
    shift_final = traj[-1]-traj[-2]
    x_final = pos_final[0] - pos_init[0]
    y_final = pos_final[1] - pos_init[1]
    deg_final = np.rad2deg(np.arctan2(shift_final[1], shift_final[0]))
    vel_init = traj[1]-traj[0]
    
    speed_traj = [np.linalg.norm(traj[i+1]-traj[i])/0.1 for i in range(len(traj)-1)]

    # print(np.linalg.norm([x_final, y_final]))
    if np.linalg.norm(pos_final - pos_init)<stop_lim: # stop during the process
      traj_type = 0
      return traj_type

    
    if np.abs(y_final) < 0.7 * lane_width:
      traj_type = 1 # straight
    
    elif y_final >= 0.7 * lane_width:
        
        if deg_final < ang_lim or y_final < 2 * lane_width:
            
            traj_type = 4 # left lc
        else:
            traj_type = 2 # left turn
    else:
        if deg_final > -1* ang_lim or y_final > -2 * lane_width:
            traj_type = 5 # right lc
        else:
            traj_type = 3 # right turn
    
    return traj_type


def _get_inter_type(data, max_agents = 32):
    # print(Interaction.overtake)
    inter_type = {"overtake" : -1, "follow" : -1, "merge" : -1, "yield" : -1, "surround" : -1, "jam" : -1}
    SAMPLE_NUM = 5
    action_step = 4
    action_dim = 1
    # future_angles = np.cumsum(self.data["future_heading"], axis=0)
    trajs = data['gt_pos']
    all_heading = data["future_heading"][:, data['agent_mask']]
    traj_each_agent = {}
    heading_each_agent = {}
    traj_each_agent, heading_each_agent = _get_all_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    # type_each_agent = _get_type_traj(data, action_step, s_rate=None, sample_num=SAMPLE_NUM)
    if len(traj_each_agent) <= 1:
        # inter_type.append(-1)
        return inter_type
    
    num_veh = len(heading_each_agent)
    num_jam = 0
    # print(num_veh)
    for i in range(num_veh):
        ego_heading = heading_each_agent[i]      
        ego_traj = traj_each_agent[i]        
        ego_type = type_traj(ego_traj)
        # print(ego_type)
        if ego_type == 0:
            num_jam += 1
        for aidx in range(len(traj_each_agent)):
            if i == aidx:
                continue
            other_traj = traj_each_agent[aidx]
            other_heading = heading_each_agent[aidx]
            other_type = type_traj(other_traj)
            is_overtake = type_overtake(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_follow = type_follow(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_merge = type_merge(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_yield = type_yield(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)
            is_surround = type_surround(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type)


            if is_overtake:
                inter_type["overtake"] = 1
            if is_follow:
                inter_type["follow"] = 1
            if is_merge:
                inter_type["merge"] = 1
            if is_yield:
                inter_type["yield"] = 1
            if is_surround:
                inter_type["surround"] = 1
           
    is_jam = type_jam(num_jam, num_veh, traj_each_agent)
    if is_jam:
        inter_type["jam"] = 1
    
    return inter_type

def type_overtake(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    LANE_WIDTH = 4.0
    
    
    if other_type != 5 and other_type !=4:
        return False
    lane_width = 4.0
    rel_dis_init, rel_pos_init = pos_rel([0.], ego_traj[0], other_traj[0])
    
    rel_dis_final, rel_pos_final = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    if rel_pos_init != 3:
        return False
    
    # if ego_traj[0, 0] == 0.0:
    #     print("------------")
    #     print(other_type)
    #     # print(ego_traj[-1])
    #     print(other_traj[0])
    #     print(other_traj[-1])
    #     abs(other_traj[-1,0] - ego_traj[-1,0])
    
    pos_front = [0, 1, 5]
    if not (rel_pos_final in pos_front) and abs(other_traj[-1,0] - ego_traj[-1,0]) > 2.0:
        return False
    
    if abs(ego_traj[0, 1] - other_traj[0, 1]) > 0.8 * LANE_WIDTH:
        return False
    
    return True

def type_follow(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    # only straight
    LANE_WIDTH = 4.0
    HEAD_LIMIT = 20.0 / 180.0 * np.pi
    if other_type != 1 or ego_type !=1:
        return False
    # print(ego_heading)
    for i in range(len(ego_heading)):
        rel_dis_init, rel_pos_init = pos_rel(ego_heading[i], ego_traj[i], other_traj[i])
        # print(rel_pos_init)
        if rel_pos_init != 3:
            return False
        if abs(ego_heading[i] - other_heading[i]) > HEAD_LIMIT:
            return False
    # rel_dis_final, rel_pos_final = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])    
    return True

def type_merge(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    LANE_WIDTH = 4.0
    rel_dis_init, rel_pos_init = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])

    if abs(ego_traj[0,1] - other_traj[0,1]) < 0.8 * LANE_WIDTH:
        return False
    if rel_pos_init != 0 and rel_pos_init != 3:
        return False
    
    # if ego_traj[0, 0] == 0.0:
    #     print("------------")
    #     print(other_type)
    #     # print(ego_traj[-1])
    #     print(other_traj[0])
    #     print(other_traj[-1])
    #     abs(other_traj[-1,0] - ego_traj[-1,0])

    if other_type != 4 and other_type != 5:
        return False
    
    return True 

def type_yield(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    if ego_type != 0:
        return False
    rel_dis_init, rel_pos_init = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    if rel_pos_init != 3:
        return False
    
    max_dis = 100
    for i in range(len(ego_heading)):
        rel_dis_init, rel_pos_init = pos_rel(ego_heading[i], ego_traj[i], other_traj[i])
        if rel_dis_init >= max_dis:
            return False
        max_dis = rel_dis_init

    return True

def type_surround(ego_traj, ego_heading, ego_type, other_traj, other_heading, other_type):
    if ego_type != 0:
        return False
    rel_dis_init, rel_pos_init = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    rel_dis_final, rel_pos_final = pos_rel(ego_heading[-1], ego_traj[-1], other_traj[-1])
    if not rel_pos_init in [2, 3, 4]:
        return False

    if not rel_pos_final in [0, 1, 5]:
        return False
    
    if not other_type in [4, 5]:
        return False

    return True

def type_jam(num_jam, num_veh, traj_each_agent):
    prob_jam = 0.8
    lim_speed = 1.0
    # is_stop = [1 if tp_agent == 0 else 0 for tp_agent in type_each_agent]
    mean_scene = num_jam / num_veh
    # print(mean_scene)
    if len(traj_each_agent) < 10:
        return False

    if mean_scene < prob_jam:
        return False
    
    return True

