from lctgen.models.utils import visualize_input_seq, visualize_map
import pickle
import torch
from torch.utils.data import DataLoader
from trafficgen.utils.data_process.agent_process import WaymoAgent
from PIL import Image

from lctgen.datasets.utils import fc_collate_fn
from lctgen.config.default import get_config
from lctgen.core.registry import registry
from lctgen.models.utils import visualize_input_seq, visualize_output_seq, visualize_map
from trafficgen.utils.typedef import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from lctgen.models.neighbor_fuse import kmeans_fuse

cfg_file = './cfgs/0.yaml'
cfg = get_config(cfg_file)
'''
model_cls = registry.get_model(cfg.MODEL.TYPE)
model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
model.eval()
'''
dataset_type = cfg.DATASET.TYPE
cfg.DATASET['CACHE'] = False
dataset = registry.get_dataset(dataset_type)(cfg, 'train')

'''
example_idx = 27 #@param {type:"slider", min:0, max:29, step:1}
dataset.data_list = [dataset.data_list[example_idx]]
'''

collate_fn = fc_collate_fn
loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory = False,
                drop_last=False, num_workers=1, collate_fn=collate_fn)

def kmeans(data, k, max_time = 10):
    init_pos_batch = []
    for j in range(data['gt_pos'][0][:, data['agent_mask'][0], :].shape[1]):
        # print("***************************************************")
        init_pos_batch.append(data['gt_pos'][0][:, data['agent_mask'][0], :][:, j, :][0])
    init_pos_batch = torch.vstack(init_pos_batch)
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
            return label        
        last_label = label
        for i in range(k):  #更新类别中心点，作为下轮迭代起始
            kpoint = init_pos_batch[label==i] 
            if i == 0:
                midpoint = kpoint.mean(0).unsqueeze(0)
            else:
                midpoint = torch.cat([midpoint, kpoint.mean(0).unsqueeze(0)], 0)
        time += 1
    return label



# type_lst = [0, 0, 0, 0, 0, 0]
def create_agent(data):
    agent = data["agent"][0].cpu().numpy()
    agent_mask = data["agent_mask"][0].cpu().numpy()
    agents = [WaymoAgent(agent[i:i+1]).length_width[0] for i in range(agent.shape[0]) if agent_mask[i]]
    return agents

for i, batch in enumerate(loader):
    data = batch
    print(data['file'])
    # print(data['traj_type'].shape)
    # type_lst = [0, 0, 0, 0, 0, 0]
    agents = create_agent(data)
    '''
    for i in agents:
        if i[0]>=2.0 or i[1]>=2.0:
            continue
            print(data['file'])
    '''
    # print(agents)
    # for i in agents:
        
    '''
    for j in range(data['gt_pos'][0][:, data['agent_mask'][0], :].shape[1]):
        # print("***************************************************")
        print(data['gt_pos'][0][:, data['agent_mask'][0], :][:, j, :])
    '''
    # feat = kmeans_fuse(data, 4, 30)
    # print(feat)
    # print("real traj: ")
    # print(data["gt_pos"][0][:, data['agent_mask'][0], :][:, 0, :])
    agents = data['agent']
    veh_type = data['traj_type'][:, data['agent_mask'][0], :].cpu().tolist()[0]
    #print(data['text'])
    
    #print(data['nei_text'])

    file_id = batch['file'][0].split(".")[0]
    file_name = "./map/" + file_id+'.png'
    map_name = "./map/" + file_id+'_map.png'
    gif_name = './map/' + file_id+'.gif'
    demo_fig = visualize_input_seq(data, save=True, filename=file_name)
    maps = visualize_map(data, save=True, path=map_name)
    # demo_gif = visualize_output_seq(data, data)
    #demo_gif[0].save(gif_name, save_all=True, append_images=demo_gif[1:])
    # break
    '''
    if i % 100 == 0:
        print(type_lst)
        print("batch "+str(i)+" finished")
    '''
'''
print(type_lst)
np_lst = np.array(type_lst)
per_lst = np_lst / np_lst.sum()
print(per_lst)
'''
print("done")
