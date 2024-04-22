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
from SearchActions import WikipediaSearch
from tqdm import tqdm

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger


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


class WikiSearchAgent(BaseAgent):
    """
    Agent to search Wikipedia content and answer questions.
    """

    def __init__(self, llm: BaseLLM, actions: List[BaseAction] = None, **kwargs):
        name = "wiki_search_agent"
        role = "Answer questions by searching Wikipedia content."
        constraint = "Generation should be simple and clear."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions or [WikipediaSearch()],
            constraint=constraint,
            logger=AgentLogger(PROMPT_DEBUG_FLAG=False),
        )


def add_few_shot_examples(agent: WikiSearchAgent = None):
    """
    Constructing the examples for agent working.
    Each example is a successful action-observation chain of an agent.
    These examples should cover all the API calls.
    """

    # Example 1: Question about Milhouse from "The Simpsons"
    task1 = "Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' character Milhouse, who was Matt Groening named after?"

    # 1. Initial thought and empty observation
    thought1_1 = "The question simplifies to 'The Simpsons' character Milhouse is named after who. I only need to search Milhouse and find who it is named after."
    act1_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought1_1})
    obs1_1 = ""

    # 2. First search action and observation
    act_params1 = {"query": "Milhouse"}
    act1_2 = AgentAct(name=WikipediaSearch().action_name, params=act_params1)
    obs1_2 = "Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening."

    # 3. Second thought to refine search
    thought1_3 = "The paragraph does not tell who Milhouse is named after, maybe I can look up 'named after'"
    act1_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought1_3})
    obs1_3 = ""

    # 4. Second search action and observation
    act_params2 = {"query": "Milhouse named after"}
    act1_4 = AgentAct(name=WikipediaSearch().action_name, params=act_params2)
    obs1_4 = "Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous."

    # 5. Final thought and finish action
    thought1_5 = "Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon."
    act1_5 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought1_5})
    obs1_5 = ""
    answer1 = "Richard Nixon"
    act1_6 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer1})
    obs1_6 = "Task Completed."

    # Adding example 1 into prompt_gen of the agent
    task_pack1 = TaskPackage(instruction=task1)
    act_obs1 = [
        (act1_1, obs1_1),
        (act1_2, obs1_2),
        (act1_3, obs1_3),
        (act1_4, obs1_4),
        (act1_5, obs1_5),
        (act1_6, obs1_6),
    ]
    agent.add_example(task=task_pack1, action_chain=act_obs1)

    # Example 2: Question about Pavel Urysohn and Leonid Levin
    # Similar steps for constructing the example with search, think, and finish actions
    task2 = "Were Pavel Urysohn and Leonid Levin known for the same type of work?"

    # 1. Initial thought and empty observation for task 2
    thought2_1 = "I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same."
    act2_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought2_1})
    obs2_1 = ""

    # 2. First search action and observation for Pavel Urysohn
    act_params2_1 = {"query": "Pavel Urysohn"}
    act2_2 = AgentAct(name=WikipediaSearch().action_name, params=act_params2_1)
    obs2_2 = "Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory."

    # 3. Second thought for searching Leonid Levin
    thought2_3 = "Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work."
    act2_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought2_3})
    obs2_3 = ""

    # 4. Second search action and observation for Leonid Levin
    act_params2_2 = {"query": "Leonid Levin"}
    act2_4 = AgentAct(name=WikipediaSearch().action_name, params=act_params2_2)
    obs2_4 = "Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist."

    # 5. Final thought and finish action for task 2
    thought2_5 = "Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work."
    act2_5 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought2_5})
    obs2_5 = ""
    answer2 = "yes"
    act2_6 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer2})
    obs2_6 = "Task Completed."

    # Adding example 2 into prompt_gen of the agent
    task_pack2 = TaskPackage(instruction=task2)
    act_obs2 = [
        (act2_1, obs2_1),
        (act2_2, obs2_2),
        (act2_3, obs2_3),
        (act2_4, obs2_4),
        (act2_5, obs2_5),
        (act2_6, obs2_6),
    ]
    agent.add_example(task=task_pack2, action_chain=act_obs2)


def run_hotpot_qa_agent(level="easy", llm_name="gpt-3.5-turbo-16k-0613"):
    """
    Test the WikiSearchAgent with a specified dataset level and LLM.
    """

    # build the search agent
    llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
    llm = get_llm_backend(llm_config)
    agent = WikiSearchAgent(llm=llm, actions=[WikipediaSearch()])
    # add several demo trajectories to the search agent for the HotPotQA benchmark
    add_few_shot_examples(agent)

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
    with open(f"data/{llm_name}_results_{level}.json", "w") as f:
        json.dump(results, f, indent=4)

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
    args = parser.parse_args()

    f1, acc = run_hotpot_qa_agent(level=args.level, llm_name=args.llm)
    print(
        f"{'+'*100}\nLLM model: {args.llm}, Dataset: {args.level}, Result: F1-Score = {f1:.4f}, Accuracy = {acc:.4f}"
    )
