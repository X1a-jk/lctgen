from lctgen.models.utils import visualize_input_seq, visualize_map

import torch
from torch.utils.data import DataLoader

from PIL import Image

from lctgen.datasets.utils import fc_collate_fn
from lctgen.config.default import get_config
from lctgen.core.registry import registry
from lctgen.models.utils import visualize_input_seq, visualize_output_seq
from lctgen.datasets.utils import traj_action_sampler
from trafficgen.utils.typedef import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import os

cfg_file = './cfgs/train.yaml'
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
loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory = False,
                drop_last=False, num_workers=1, collate_fn=collate_fn)

sampler = traj_action_sampler(dataset, 8)
print(sampler)


# type_lst = [0, 0, 0, 0, 0, 0]
for i, batch in enumerate(loader):
    data = batch
    # print(data['file'])
    # type_lst = [0, 0, 0, 0, 0, 0]
    '''
    for j in range(data['gt_pos'][0][:, data['agent_mask'][0], :].shape[1]):
        print("***************************************************")
        print(data['gt_pos'][0][:, data['agent_mask'][0], :][:, j, :])
    '''
    # print("real traj: ")
    # print(data["gt_pos"][0][:, data['agent_mask'][0], :][:, 0, :])
    agents = data['agent']
    veh_type = data['traj_type'][:, data['agent_mask'][0], :].cpu().tolist()[0]

    file_id = batch['file'][0].split(".")[0]
    file_name = "./map/" + file_id+'.png'
    gif_name = './map/' + file_id+'.gif'
    # demo_fig = visualize_input_seq(data, save=True, filename=file_name)
    # demo_gif = visualize_output_seq(data, data)
    # demo_gif[0].save(gif_name, save_all=True, append_images=demo_gif[1:])
    break
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
