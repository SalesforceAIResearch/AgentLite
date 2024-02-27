from typing import List

import os

from webshop_agents import build_act_agent
from webshop_actions import webshop_env

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger


# =============================== start of webshop agent designing =============================== #

REWARD_LOG_FILE = "reward.txt"
llm_name = "gpt-3.5-turbo-16k"
temperature = 0.0
llm_config_dict = {"llm_name": llm_name, "temperature": temperature}
agent_logger = AgentLogger(PROMPT_DEBUG_FLAG=False)

# test 
def evalute(id: int):
    env_idx = f"fixed_{id}"
    
    agent = build_act_agent(session_idx=env_idx, llm_config_dict=llm_config_dict, logger=agent_logger)
    # reset the env
    action = "reset[]"
    observation, reward, done, asins, clickable = webshop_env.step(env_idx, action)

    task = webshop_env.goal
    task_package = TaskPackage(instruction=task)
    agent(task_package)
    reward = webshop_env.reward 
    sub_reward = webshop_env.sub_reward
    return reward, sub_reward, task

# using this function to rerun the evaluation if breaks
def get_runned_ids(file_path):
    try:
        with open(file_path, 'r') as file:
            runned_ids = [int(line.split()[0]) for line in file]
        return runned_ids
    except FileNotFoundError:
        print("The file was not found.")
        return None
    except ValueError:
        print("The last item in the last line is not a valid number.")
        return None

rewards = []
all_task_ids = list(range(0, 252))

runned_ids = get_runned_ids(REWARD_LOG_FILE)
if runned_ids is None:
    evalute_ids = all_task_ids
else:
    evalute_ids = [id for id in all_task_ids if id not in runned_ids]

# running webshop evaluation
with open(REWARD_LOG_FILE, 'a') as f:
    for i in evalute_ids:
        reward, subreward, task = evalute(i)
        rewards.append(reward)
        reward_str = f"""{i}\t{task}\t{subreward}\t{reward}\n"""
        f.write(reward_str)