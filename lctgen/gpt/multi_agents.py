import abc
import pathlib
import os
import openai

from autogen import ConversableAgent

import sys
sys.path.append("/home/ubuntu/xiajunkai/lctgen/")
# from model_old import HDGT_model#_new
import importlib
model_module = importlib.import_module("lctgen")
from lctgen.inference.utils import output_formating_cot, map_retrival, get_map_data_batch
folder = os.path.dirname(__file__)

# org_path = os.path.join(folder, 'api.org')
# api_path = os.path.join(folder, 'api.key')

# openai.organization = open(org_path).read().strip()
# openai.api_key = open(api_path).read().strip()

from lctgen.core.basic import BasicLLM
from lctgen.core.registry import registry
from lctgen.config.agent_cfg import Config, get_config 
from lctgen.inference.utils import load_all_map_vectors


OPENAI_API_KEY = ""
OPENAI_BASE_URL = "https://integrate.api.nvidia.com/v1" #"https://api.openai-proxy.com/v1/"

class Basic_Agent():
    def __init__(self, config, name="default"):
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        # self.system_prompt_path = None
        sys_prompt_path = None #"lctgen/gpt/prompts/attr_ind_motion/"
        query_prompt_path = None
        self.sys_prompt = None # open(sys_prompt_path).read().strip()
        self.que_prompt = None # open(sys_prompt_path).read().strip()
        self.name = name

        # self.create_agent()

    def prepare_prompt(self, query, base_prompt):
        extended_prompt = base_prompt.replace("INSERT_QUERY_HERE", query)
        return extended_prompt
    
    def create_agent(self, termination_msg=None):
        self.agent = ConversableAgent(
        self.name,
        system_message = self.sys_prompt,
        is_termination_msg = termination_msg,
        llm_config={"config_list": [{"model": self.model, "api_key": self.api_key, "base_url": self.base_url}]},
        code_execution_config=False,  # Turn off code execution, by default it is off.
        function_map=None,  # No registered functions, by default it is None.
        human_input_mode="NEVER",  # Never ask for human input.        
        )
    
    def llm_query(self, extended_prompt, role="user"):
        
        reply = self.agent.generate_reply(messages=[{"content": extended_prompt, "role": role}])
        return reply

    def process_request(self, query):
        
        extended_prompt = self.prepare_prompt(query, self.que_prompt)
        reply = self.llm_query(extended_prompt)
        return reply
        


    def post_process(self, response):
        return response
    
    

    

class Analyser(Basic_Agent):
    def __init__(self, config, name="analyser_0"):
        super().__init__(config, name)
        sys_prompt_path = "lctgen/gpt/prompts/attr_ind_motion/sys_analyzer.prompt"
        query_prompt_path = "lctgen/gpt/prompts/attr_ind_motion/que_analyzer.prompt"
        self.sys_prompt = open(sys_prompt_path).read().strip()
        self.que_prompt = open(query_prompt_path).read().strip()
        # print(f"{self.sys_prompt=}")
        # print(f"{self.que_prompt=}")  
        termination_msg = lambda msg: ("Mapping Vector" in msg["content"] or "Actor Vector" in msg["content"])
        self.create_agent(termination_msg)

    def analyze(self, guidance):
        global map_type
        global total_num
        global agent_num
        global des_idx
        result = self.process_request(guidance)
        print("reply: ")
        print(result)
        lines = result.split('\n')
        veh_description = []
        ped_description = []
        bike_description = []
        
        for idx, line in enumerate(lines):
            if "'map type'" in line:
                map_type = line.split(": ")[-1]
                continue
            if "'total number'" in line:
                total_num = int(line.split(": ")[-1])
                continue
            if "'agent number'" in line:
                agent_num = eval(line.split(": ")[-1])
                des_idx = idx
                continue

        for i in range(des_idx+1, len(lines)):
            if lines[i].startswith("- 'Vehicle"):
                veh_description.append(lines[i].split(": ")[-1])
                continue
            if lines[i].startswith("- 'Pedestrian"):
                ped_description.append(lines[i].split(": ")[-1])
                continue
            if lines[i].startswith("- 'Bicycle"):
                bike_description.append(lines[i].split(": ")[-1])
                continue
        
        self.map_type = map_type
        self.agent_num = agent_num
        self.total_num = total_num
        self.veh_description = veh_description
        self.ped_description = ped_description
        self.bike_description = bike_description
        
        # print(f"{map_type=}")
        # print(f"{agent_num=}")
        # print(f"{total_num=}")
        # print(f"{veh_description=}")
        # print(f"{ped_description=}")
        # print(f"{bike_description=}")
        self.initiate_agents()
            
    def initiate_agents(self):
        retriever_config = get_config() #default config
        agent_config = get_config() #default config
        self.map_retriver = Retriever(retriever_config, "map_retriever")
        agent_list = []
        agent_names = ["Vehicle", "Pedestrian", "Bicycle"]

        for idx, num_type in enumerate(self.agent_num):
            # idx -> 0: veh, 1:ped, 2:bike
            for idx_vh in range(int(num_type)):
                agent_name_temp = agent_names[idx]+f"_{idx_vh}"
                agent_temp = Executor(agent_config, agent_name_temp)
                agent_list.append((idx, idx_vh, agent_temp))

        self.agent_list = agent_list
                
        print(f"{self.agent_list=}")

        


    def pass_msg_to_agent(self, agent_temp, idx, idx_vh):
        if idx == 0:
            msg_temp = self.veh_description[idx_vh]
        elif idx == 1:
            msg_temp = self.ped_description[idx_vh]
        else:
            msg_temp = self.bike_description[idx_vh]
        agent_result = self.agent.initiate_chat(agent_temp.agent, message = msg_temp, summary_method="reflection_with_llm", max_turns=2)

        return agent_result

    def generate_map(self):
        chat_result = self.agent.initiate_chat(self.map_retriver.agent, message=f"Generate a map containing {self.map_type}", summary_method="reflection_with_llm", max_turns=2)
        useful_info = chat_result.chat_history[-1]['content'].split('\n')
        global map_idx_in_info
        for idx, line in enumerate(useful_info):
            if "Mapping Vector" in line:
                map_idx_in_info = idx
                break

        map_code = eval(useful_info[map_idx_in_info+1].split(": ")[-1]) # map_code = eval(useful_info[map_idx_in_info+1].split(": ")[-1])

        self.map_code = map_code

        


    def arrange_tasks(self):
        replys = []
        for agent_info in self.agent_list:
            idx, idx_vh, agent_temp = agent_info
            reply_temp = self.pass_msg_to_agent(agent_temp, idx, idx_vh)
            replys.append(reply_temp.summary)
            
        print(replys)

class Retriever(Basic_Agent):
    def __init__(self, config, name = "retriever_0"):
        super().__init__(config, name)
        sys_prompt_path = "lctgen/gpt/prompts/attr_ind_motion/sys_retriever.prompt"
        query_prompt_path = "lctgen/gpt/prompts/attr_ind_motion/que_retriever.prompt"
        self.sys_prompt = open(sys_prompt_path).read().strip()
        self.que_prompt = open(query_prompt_path).read().strip()
        # print(f"{self.sys_prompt=}")
        # print(f"{self.que_prompt=}")  
        self.create_agent()

    def lctgen_get_map(self, map_vector, map_cfg):
        map_data_file = "/home/ubuntu/xiajunkai/lctgen/data/map.npy"
        map_vecs, map_ids = load_all_map_vectors(map_data_file)

        sorted_idx = map_retrival(map_vector, map_vecs)[:1]
        map_id = map_ids[sorted_idx[0]]

        map_id = '1_1578.pkl 10'
        #load map data
        batch = get_map_data_batch(map_id, map_cfg)
        return batch
    
    def generate_code(self, map_type):
        map_information = self.process_request(map_type)
        print("map_code")
        print(map_information)


map_data_file = "/home/ubuntu/xiajunkai/lctgen/data/map.npy"
map_vecs, map_ids = load_all_map_vectors(map_data_file)

class Executor(Basic_Agent):
    def __init__(self, config, name = "agent_0"):
        super().__init__(config, name)
        sys_prompt_path = "lctgen/gpt/prompts/attr_ind_motion/sys_agent.prompt"
        query_prompt_path = "lctgen/gpt/prompts/attr_ind_motion/que_agent.prompt"
        self.sys_prompt = open(sys_prompt_path).read().strip()
        self.que_prompt = open(query_prompt_path).read().strip()
        # print(f"{self.sys_prompt=}")
        # print(f"{self.que_prompt=}")  
        self.create_agent()

    def generate_code(self, guidance):
        agent_information = self.process_request(guidance)
        



class Evaluator(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)

def translate_guidance(guidance: str) -> list:
    pass

if __name__ == "__main__":
    # agent = ConversableAgent(
    #     "chatbot",
    #     llm_config={"config_list": [{"model": "gpt-4", "api_key": OPENAI_API_KEY, "base_url": OPENAI_BASE_URL}]},
    #     code_execution_config=False,  # Turn off code execution, by default it is off.
    #     function_map=None,  # No registered functions, by default it is None.
    #     human_input_mode="NEVER",  # Never ask for human input.
    # )

    # reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])
    # print(reply)
    config = get_config()

    analyser = Analyser(config, "analyzer")
    analyser.analyze("Generate a scenario with six agents including two vehicles, two pedestrians and two bicycles.")
    analyser.generate_map()
    analyser.arrange_tasks()
    # print(analyser.process_request("Generate a scenario with eight agents including two vehicles, a pedestrian and a bicycle."))

