import torch
import numpy as np
from trafficgen.utils.data_process.agent_process import WaymoAgent
from trafficgen.utils.visual_init import draw
from torch.utils.data.dataloader import default_collate
from torch.utils.data import WeightedRandomSampler

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

def fc_collate_fn(batch):
  result_batch = {}
  
  for key in batch[0].keys():
    if 'other' in key or 'center_info' in key:
      result_batch[key] = [item[key] for item in batch]
    else:
      result_batch[key] = default_collate([item[key] for item in batch])
  
  return result_batch

def traj_action_sampler(dataset):
  type_frequency = [0.42270312, 0.50473542, 0.02286027, 0.020658, 0.01507475, 0.01396844] #stop, straigt, left-turn, right-turn, left-change-lane, right-change-lane
  weight_frequency = [1.0 / t for t in type_frequency]
  weights_type = []
  for i in range(len(dataset)):
    data_temp = dataset[i]
    agent_valid = 0
    wgt_temp = 0.0
    for tp in data_temp['traj_type']:
      if tp[0] == -2:
        continue
      agent_valid += 1
      wgt_temp += weight_frequency[tp[0]]
    weights_type.append(wgt_temp / agent_valid)
  print(len(dataset))
  print(len(weights_type))
  print(weights_type)
  return 0