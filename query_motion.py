import torch
from torch.utils.data import DataLoader
import pickle
from PIL import Image

from lctgen.datasets.utils import fc_collate_fn
from lctgen.config.default import get_config
from lctgen.core.registry import registry
from lctgen.models.utils import visualize_input_seq, visualize_output_seq

from trafficgen.utils.typedef import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import os

def vis_decode(batch, ae_output):
    img = visualize_output_seq(batch, output=ae_output[0], pool_num=1)
    return img

def vis_stat(batch, ae_output):
    img = visualize_input_seq(batch, agents=ae_output[0]['agent'], traj=ae_output[0]['traj'])
    return Image.fromarray(img)

cfg_file = './cfgs/inference.yaml'
cfg = get_config(cfg_file)

model_cls = registry.get_model(cfg.MODEL.TYPE)
model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
model.eval()

dataset_type = cfg.DATASET.TYPE
cfg.DATASET['CACHE'] = False
dataset = registry.get_dataset(dataset_type)(cfg, 'train')

example_idx = 27 #@param {type:"slider", min:0, max:29, step:1}

'''
dataset.data_list = [dataset.data_list[example_idx]]
collate_fn = fc_collate_fn
loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory = False,
                drop_last=False, num_workers=1, collate_fn=collate_fn)

for batch in loader:
    break
'''

from lctgen.inference.utils import output_formating_cot, map_retrival, get_map_data_batch

def traj2act(dataloader, exp_id):
    base_path = "./traj2act_new/"
    for batch in loader:
        text = batch['text'][batch['agent_mask']] #27*13
        traj = batch['traj'].squeeze(0)[:, batch['agent_mask'][0], :] #50*27*2
        num_agent, dim_agent = text.shape
        for i in range(num_agent):
            txt_temp = text[i, :].cpu().numpy()[4:].astype(np.int32).tolist()
            trj_temp = traj[:, i, :].squeeze(1).cpu().numpy()
            x = trj_temp[:, 0].reshape((-1,1))
            y = trj_temp[:, 1].reshape((-1,1))
            plt.figure()
            plt.axis('equal')
            plt.plot(x, y,  marker='*')
            plt.xlim([-80, 80])
            plt.ylim([-80, 80])
            file_name = base_path+str(exp_id)+"/"+str(i)+"_"+str(txt_temp)+".jpg"
            # plt.savefig(file_name)
            # break
       #break
        




def gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids):

    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32
    agent_vector, map_vector, event_vector = output_formating_cot(llm_text)
    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    event_dim = len(event_vector[0])

    type_vector = [it[-1] for it in agent_vector]
    agent_vector = [it[:-1] + [-1] for it in agent_vector]
    
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)
    event_vector = event_vector + [[-1]*event_dim] * (MAX_AGENT_NUM - agent_num)
    
    # retrive map from map dataset
    sorted_idx = map_retrival(map_vector, map_vecs)[:1]
    map_id = map_ids[sorted_idx[0]]
    # map_id = '0_343.pkl 100'
    # map_id = '0_288.pkl 100'
    #load map data
    batch = get_map_data_batch(map_id, cfg)
    type_len = batch['traj_type'].shape[1]
    for i in range(type_len):
        if i<len(type_vector):
            batch['traj_type'][0, i, 0] = type_vector[i]
        else:
            batch['traj_type'][0, i, 0] = -2
    # inference with LLM-output Structured Representation
#    print(batch['text'])   
    
    batch['text'] = torch.tensor(agent_vector, dtype=batch['text'].dtype, device=model.device)[None, ...]
    event_tensor = torch.tensor(event_vector, dtype=batch['nei_text'][1].dtype, device=model.device)[None, ...]
    batch['text'] = batch['text'][:, :, :-1]
    batch['nei_text'] = [batch['nei_text'][0], event_tensor]
    b, d, _ = batch['text'].shape
    padding = -1 * torch.ones((b, d, 1), device=model.device)
#    batch['text'] = torch.cat((padding, batch['text']), dim=-1)
    b_2, d_2, _ = batch['nei_text'][1].shape
    padding_2 = -1 * torch.ones((b_2, d_2, 1), device=model.device)
    # batch['nei_text'][1] = torch.cat((batch['nei_text'][1],padding_2), dim=-1)
    batch['agent_mask'] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), \
            dtype=batch['agent_mask'].dtype, device=model.device)[None, ...]
#    print(batch['text'])
#    with open('6_2360.pickle', 'rb') as file:
#        batch = pickle.load(file)
    
    print(batch['file'])
    for k in batch.keys():
        if type(batch[k])==torch.Tensor:
            batch[k] = batch[k].to(model.device)

    exceed = True
    i = 0
    while exceed and i<10:
        print(f"repeat times {i}")
        model_output = model.forward(batch, 'val')['text_decode_output']
        output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)
        # return "finished"
        exceed = output_scene[0]['exceed']
        print(exceed)
        i+=1

    

    return vis_decode(batch, output_scene), vis_stat(batch, output_scene)

from lctgen.inference.utils import load_all_map_vectors

map_data_file = "/home/ubuntu/xiajunkai/lctgen/data/map.npy"
map_vecs, map_ids = load_all_map_vectors(map_data_file)

from lctgen.core.registry import registry
from lctgen.config.default import get_config

llm_cfg = get_config('./lctgen/gpt/cfgs/attr_ind_motion/traj.yaml') # new.yaml')
llm_model = registry.get_llm('codex')(llm_cfg)
# print(llm_model.device)

import openai

'''
openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"
'''

openai.api_key = ""
openai.base_url = "https://api.openai-proxy.com/v1/"


def infer_result(response):
    if "stop" in response:
        return 0
    if "straight" in response:
        return 1
    if "left" in response:
        if "turn" in response:
            return 2
        elif "change" in response:
            return 4
    if "right" in response:
        if "turn" in response:
            return 3
        elif "change" in response:
            return 5
    
    return -1
        


from lctgen.datasets.utils import fc_collate_fn
loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False, drop_last=False, num_workers=1, collate_fn=fc_collate_fn)
tot = 0
success = 0
for i, batch in enumerate(loader):
    data = batch
    gt_type_batch = data["traj_type"][:, data['agent_mask'][0], :].cpu().tolist()
    
    for j in range(data['gt_pos'][0][:, data['agent_mask'][0, :]].shape[1]):
        query = [(round(i[0],2), round(i[1],2)) for i in data['gt_pos'][0][:, data['agent_mask'][0, :]][:, j, :][::5, :].cpu().tolist()]
        gt_type = int(gt_type_batch[0][j][0])

        llm_result = llm_model.forward(str(query))
        type = infer_result(llm_result)
        if type == gt_type:
            success += 1
        
        tot += 1
    
        if tot == 100:
            print(success)
            print(tot)
            print(f"success_rate: {success/tot}")

print(success)
print(tot)
print(f"success_rate: {success/tot}")
        

# query = "Genarate an overtaking scene"

# print("query: ")
# print(query)




# llm_result = llm_model.forward(query)
