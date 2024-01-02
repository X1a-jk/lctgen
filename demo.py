import torch
from torch.utils.data import DataLoader

from PIL import Image

from lctgen.datasets.utils import fc_collate_fn
from lctgen.config.default import get_config
from lctgen.core.registry import registry
from lctgen.models.utils import visualize_input_seq, visualize_output_seq

from trafficgen.utils.typedef import *


def vis_decode(batch, ae_output):
    img = visualize_output_seq(batch, output=ae_output[0], pool_num=1)
    return img


cfg_file = './cfgs/demo_inference.yaml'
cfg = get_config(cfg_file)

model_cls = registry.get_model(cfg.MODEL.TYPE)
model = model_cls.load_from_checkpoint(cfg.LOAD_CHECKPOINT_PATH, config=cfg, metrics=[], strict=False)
model.eval()

dataset_type = cfg.DATASET.TYPE
cfg.DATASET['CACHE'] = False
dataset = registry.get_dataset(dataset_type)(cfg, 'train')

example_idx = 27 #@param {type:"slider", min:0, max:29, step:1}

dataset.data_list = [dataset.data_list[example_idx]]
collate_fn = fc_collate_fn
loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory = False,
                drop_last=False, num_workers=1, collate_fn=collate_fn)
for batch in loader:
    break

from lctgen.inference.utils import output_formating_cot, map_retrival, get_map_data_batch

def gen_scenario_from_gpt_text(llm_text, cfg, model, map_vecs, map_ids):

    # format LLM output to Structured Representation (agent and map vectors)
    MAX_AGENT_NUM = 32
    agent_vector, map_vector = output_formating_cot(llm_text)

    agent_num = len(agent_vector)
    vector_dim = len(agent_vector[0])
    agent_vector = agent_vector + [[-1]*vector_dim] * (MAX_AGENT_NUM - agent_num)

    # retrive map from map dataset
    sorted_idx = map_retrival(map_vector, map_vecs)[:1]
    map_id = map_ids[sorted_idx[0]]

    # load map data
    batch = get_map_data_batch(map_id, cfg)

    # inference with LLM-output Structured Representation
    batch['text'] = torch.tensor(agent_vector, dtype=batch['text'].dtype, device=model.device)[None, ...]
    batch['agent_mask'] = torch.tensor([1]*agent_num + [0]*(MAX_AGENT_NUM - agent_num), \
            dtype=batch['agent_mask'].dtype, device=model.device)[None, ...]
    # batch['device'] = model.device    
    for k in batch.keys():
        if type(batch[k])==torch.Tensor:
            batch[k] = batch[k].to(model.device)
    model_output = model.forward(batch, 'val')['text_decode_output']
    output_scene = model.process(model_output, batch, num_limit=1, with_attribute=True, pred_ego=True, pred_motion=True)

    return vis_decode(batch, output_scene)

from lctgen.inference.utils import load_all_map_vectors

map_data_file = './data/demo/waymo/demo_map_vec.npy'
map_vecs, map_ids = load_all_map_vectors(map_data_file)

from lctgen.core.registry import registry
from lctgen.config.default import get_config

llm_cfg = get_config('./lctgen/gpt/cfgs/attr_ind_motion/non_api_cot_attr_20m.yaml')
llm_model = registry.get_llm('codex')(llm_cfg)
# print(llm_model.device)

import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

query = 'The scene has multiple lanes. Two cars are driving on the center lane. Ego car is driving slowly, and the other car accelerates to overtake ego car.'  # @param {type:"string"}

print("query: ")
print(query)

llm_result = llm_model.forward(query)

print('LLM inference result:')
print(llm_result)

cfg_file = 'cfgs/inference.yaml'
cfg = get_config(cfg_file)

img_list = gen_scenario_from_gpt_text(llm_result, cfg, model, map_vecs, map_ids)
print("img_list generated")
img_list[0].save("demo_ot.gif", save_all=True, append_images=img_list[1:])

