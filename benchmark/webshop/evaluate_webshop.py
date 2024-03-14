from typing import List

import os
import argparse

from webshop_agents import WebshopAgent
from webshop_actions import webshop_env
from webshop_multiagent import bolaa_webagent

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger



# =============================== start of webshop agent designing =============================== #

def evalute(idx: int, llm_name="gpt-3.5-turbo-16k-0613", agent_arch="react", PROMPT_DEBUG_FLAG=False):
    if llm_name in ["xlam", "xlam_v2"]:
        llm_config = LLMConfig(
            {
                "llm_name": llm_name, 
                "temperature": 0.0, 
                "base_url": "http://localhost:8000/v1",
                "openai_api_key": "EMPTY"
            }
        )
    else:
        llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
    llm = get_llm_backend(llm_config)
    env_idx = f"fixed_{idx}"
    if agent_arch in ["bolaa"]:
        agent = bolaa_webagent(session_idx=env_idx, env=webshop_env, llm=llm, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        task = agent.goal
        agent.run()
    else:
        # reset the env first if not using bolaa agent
        action = "reset[]"
        webshop_env.step(env_idx, action)
        agent = WebshopAgent(session_idx=env_idx, llm=llm, agent_arch=agent_arch, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        task = webshop_env.goal
        print(f"Task: {task}")
        task_package = TaskPackage(instruction=task)
        agent(task_package)
    reward = webshop_env.reward
    sub_reward = webshop_env.sub_reward
    return reward, sub_reward, task

# using this function to rerun the evaluation if breaks
def get_runned_ids(file_path):
    try:
        with open(file_path, "r") as file:
            runned_ids = [int(line.split()[0]) for line in file]
        return runned_ids
    except FileNotFoundError:
        print("The file was not found.")
        return None
    except ValueError:
        print("The last item in the last line is not a valid number.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Search Agent on the webshop Benchmark"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-3.5-turbo-16k-0613",
        help="Name of the language model",
    )
    parser.add_argument(
        "--agent_arch",
        type=str,
        choices=["react", "act", "planact", "planreact", "zs", "zst", "bolaa"],
        default="react",
        help="agent reasoning type",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug flag",
    )
    args = parser.parse_args()
    rewards = []
    all_task_ids = list(range(0, 252))
    REWARD_LOG_FILE = f"{args.llm}_{args.agent_arch}_results_webshop.csv"
    runned_ids = get_runned_ids(REWARD_LOG_FILE)
    if runned_ids is None:
        evalute_ids = all_task_ids
    else:
        evalute_ids = [id for id in all_task_ids if id not in runned_ids]

    # running webshop evaluation
    with open(REWARD_LOG_FILE, "a") as f:
        for i in evalute_ids:
            reward, subreward, task = evalute(i, llm_name=args.llm, agent_arch=args.agent_arch, PROMPT_DEBUG_FLAG=args.debug)
            rewards.append(reward)
            reward_str = f"""{i}\t{task}\t{subreward}\t{reward}\n"""
            f.write(reward_str)

