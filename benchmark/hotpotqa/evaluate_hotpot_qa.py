import argparse
import json
import os
import re
import string
from collections import Counter
from typing import List

import joblib
import numpy as np
import requests
from tqdm import tqdm
from SearchActions import WikipediaSearch
from hotpotagents import WikiSearchAgent 


from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger



def download_file(url, filename):
    """
    Download a file from a URL and save it locally.
    """
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {filename}")


def load_hotpot_qa_data(level):
    """
    Load HotpotQA data for a given level. If data doesn't exist, download it.
    """
    file_path = f"./data/{level}.joblib"
    data_url = (
        f"https://github.com/salesforce/BOLAA/raw/main/hotpotqa_run/data/{level}.joblib"
    )

    if not os.path.exists(file_path):
        print(f"{level} data not found, downloading...")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        download_file(data_url, file_path)
    # joblib requires python 3.10 or higher
    return joblib.load(file_path)


def normalize_answer(s):
    """
    Normalize answers for evaluation.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Compute the F1 score between prediction and ground truth answers.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def run_hotpot_qa_agent(level="easy", llm_name="gpt-3.5-turbo-16k-0613", agent_arch="react", PROMPT_DEBUG_FLAG=False):
    """
    Test the WikiSearchAgent with a specified dataset level and LLM.
    """

    # build the search agent
    llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
    
    if llm_name == "xlam_v2":
        llm_config = LLMConfig(
            {
                "llm_name": llm_name,
                "temperature": 0.0,
                "base_url": "http://localhost:8000/v1",
                "api_key": "EMPTY",
            }
        )
    llm = get_llm_backend(llm_config)
    agent = WikiSearchAgent(llm=llm, agent_arch=agent_arch, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
    # add several demo trajectories to the search agent for the HotPotQA benchmark
    hotpot_data = load_hotpot_qa_data(level)
    hotpot_data = hotpot_data.reset_index(drop=True)
    task_instructions = [
        (row["question"], row["answer"]) for _, row in hotpot_data.iterrows()
    ]
    f1_list, correct, results = [], 0, {}
    for test_task, answer in tqdm(task_instructions, desc="Processing"):
        test_task_pack = TaskPackage(instruction=test_task)

        try:
            response = agent(test_task_pack)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            response = "Cannot find the answer."

        f1, _, _ = f1_score(response, answer)
        f1_list.append(f1)
        correct += int(response == answer)
        results[test_task] = (response, answer)

        avg_f1 = np.mean(f1_list)
        acc = correct / len(task_instructions)
        
        dump_str = f"{test_task}\t{answer}\t{response}\t{f1:.4f}\t{acc:.4f}\n"
        with open(f"data/{agent_arch}_{llm_name}_results_{level}.csv", "a") as f:
            f.write(dump_str)

    return avg_f1, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Search Agent on the HotPotQA Benchmark"
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty level of the dataset.",
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
        choices=["react", "act", "planact", "planreact", "zs", "zst"],
        default="react",
        help="agent reasoning type",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug flag",
    )
    args = parser.parse_args()

    f1, acc = run_hotpot_qa_agent(level=args.level, llm_name=args.llm, agent_arch=args.agent_arch, PROMPT_DEBUG_FLAG=args.debug)
    print(
        f"{'+'*100}\nLLM model: {args.llm}, Dataset: {args.level}, Result: F1-Score = {f1:.4f}, Accuracy = {acc:.4f}"
    )
