import torch
import numpy as np


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

        pos_agent = pos_agent[::sample_rate][:sample_num]
        traj_each_agent.update({aix: pos_agent})
        heading_agent = heading_agent[valid_mask]
        heading_agent = heading_agent[::sample_rate][:sample_num].reshape((-1,1))
        heading_each_agent.update({aix: heading_agent})

    return traj_each_agent, heading_each_agent

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
    deg_other = np.arctan2(pos_rel[1], pos_rel[0]) if pos_rel[0] != 0 else np.pi/2
    deg_rel = deg_other - ego_heading
    if deg_rel > np.pi:
        deg_rel -= 2 * np.pi
    elif deg_rel < -1 * np.pi:
        deg_rel += 2*np.pi
        
    if deg_rel < np.pi/9 and deg_rel > -1 * np.pi / 9:
        ang = 0
    elif deg_rel <= -1 * np.pi / 9 and deg_rel > -1 * np.pi / 2:
        ang = 1
    elif  deg_rel <= -1 * np.pi / 2 and deg_rel > -8 * np.pi/9:
        ang = 2
    elif deg_rel >= np.pi/9 and deg_rel < np.pi/2:
        ang = 5
    elif deg_rel >= np.pi/2 and deg_rel < 8*np.pi/9:
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

    
