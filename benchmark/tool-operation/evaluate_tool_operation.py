import argparse
import os

from tool_operation_utils import get_data
from tool_operation_agents import TodoAgent, SheetAgent 
from todo_env import TodoEnv
from sheet_env import SheetEnv

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger

# LAM_URL = os.environ["LAM_URL"]
LAM_URL = "http://"

def evaluate(data_name: str, idx: int, llm_name="gpt-4", agent_arch="react", PROMPT_DEBUG_FLAG=False):
    if llm_name in ["xlam", "xlam_v2"]:
        llm_config = LLMConfig(
            {
                "llm_name": llm_name, 
                "temperature": 0.0, 
                "base_url": LAM_URL,
                "api_key": "EMPTY"
            }
        )
    else:
        llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
    llm = get_llm_backend(llm_config)
    dataset_i = get_data(idx, data_name)
    tool_type = dataset_i["tool"]
    if tool_type == "todo":
        env = TodoEnv(dataset=dataset_i)
        agent = TodoAgent(env=env, llm=llm, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        task = dataset_i["goal"]
        print(f"Task: {task}")
        task_package = TaskPackage(instruction=task)
        response = agent(task_package)
        reward = env.reward
        print(f"Reward: {reward}")
        return reward, task, response
    elif tool_type == "sheet":
        env = SheetEnv(dataset=dataset_i)
        agent = SheetAgent(env=env, llm=llm, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        task = dataset_i["goal"]
        print(f"Task: {task}")
        task_package = TaskPackage(instruction=task)
        response = agent(task_package)
        reward = env.reward
        print(f"Reward: {reward}")
        return reward, task, response


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
        default="gpt-4",
        help="Name of the language model",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="tool-operation",
        help="tool dataset name",
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
    REWARD_LOG_FILE = f"{args.llm}_{args.agent_arch}_{args.data_name}_results_tool_operation.csv"
    all_task_ids = list(range(0, 60))
    finished_ids = get_runned_ids(REWARD_LOG_FILE)
    if finished_ids is None:
        evaluate_ids = all_task_ids
    else:
        evaluate_ids = [id for id in all_task_ids if id not in finished_ids]

    with open(REWARD_LOG_FILE, "a") as f:
        for idx in evaluate_ids:
            reward, task, response = evaluate(data_name=args.data_name, idx=idx, llm_name=args.llm, agent_arch=args.agent_arch, PROMPT_DEBUG_FLAG=args.debug)
            print(f"Task: {task}")
            print(f"Response: {response}")
            print(f"Reward: {reward}")
            print("=====================================")
            reward_str = f"""{idx}\t{task}\t{reward}\t{response}\n"""
            f.write(reward_str)
    
    # calculate the average reward
    # read the file and calculate the average reward
    with open(REWARD_LOG_FILE, "r") as f:
        lines = f.readlines()
        rewards = [float(line.split('\t')[2]) for line in lines]
    
    avg_reward = sum(rewards) / len(rewards)
    print(f"The average reward is: {avg_reward}")

        