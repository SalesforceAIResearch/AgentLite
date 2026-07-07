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
from hotpotagents import WikiSearchAgent 


from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger


DEFAULT_RAG_INDEX_DIR = "/home/xuzm/data/ICCAD2026/rag_cag_algorithm/TurboRAG/emb_wiki_bz2_nomic"
DEFAULT_RAG_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_RAG_EMBEDDING_DIM = 256
DEFAULT_RAG_DEVICE = "cpu"
DEFAULT_RAG_TEXT_LOOKUP = "cache"


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


def run_hotpot_qa_agent(
    level="easy",
    llm_name="gpt-3.5-turbo-16k-0613",
    agent_arch="react",
    PROMPT_DEBUG_FLAG=False,
    search_mode="online",
    rag_index_dir=DEFAULT_RAG_INDEX_DIR,
    rag_top_k=5,
    rag_strategy="cosine",
    rag_embed_model=DEFAULT_RAG_EMBED_MODEL,
    rag_embedding_dim=DEFAULT_RAG_EMBEDDING_DIM,
    rag_cache_dir=None,
    rag_device=DEFAULT_RAG_DEVICE,
    rag_text_lookup=DEFAULT_RAG_TEXT_LOOKUP,
    max_tasks=None,
):
    """
    Test the WikiSearchAgent with a specified dataset level and LLM.
    """

    # build the search agent
    llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
    # running xlam 
    if llm_name in ["xlam", "xlam_v2"]:
        llm_config = LLMConfig(
            {
                "llm_name": llm_name, 
                "temperature": 0.0, 
                "base_url": "http://localhost:8000/v1",
                "api_key": "EMPTY"
            }
        )
    # running glm5.2 via siliconflow (OpenAI-compatible endpoint)
    if llm_name in ["glm5.2", "glm-5.2"]:
        llm_config = LLMConfig(
            {
                "llm_name": "zai-org/GLM-5.2",
                "temperature": 0.0,
                "base_url": "https://api.siliconflow.cn/v1",
                "api_key": os.environ.get("SILICONFLOW_API_KEY", "EMPTY"),
                "max_tokens": 1024,
            }
        )
        # GLM-5.2 is a chat model, but not in OPENAI_CHAT_MODELS, so route
        # explicitly to LangchainChatModel instead of the default completion path.
        from agentlite.llm.agent_llms import LangchainChatModel
        llm = LangchainChatModel(llm_config)
    else:
        llm = get_llm_backend(llm_config)
    agent = WikiSearchAgent(
        llm=llm,
        agent_arch=agent_arch,
        PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG,
        search_mode=search_mode,
        rag_index_dir=rag_index_dir,
        rag_top_k=rag_top_k,
        rag_strategy=rag_strategy,
        rag_embed_model=rag_embed_model,
        rag_embedding_dim=rag_embedding_dim,
        rag_cache_dir=rag_cache_dir,
        rag_device=rag_device,
        rag_text_lookup=rag_text_lookup,
    )
    # add several demo trajectories to the search agent for the HotPotQA benchmark
    hotpot_data = load_hotpot_qa_data(level)
    hotpot_data = hotpot_data.reset_index(drop=True)
    task_instructions = [
        (row["question"], row["answer"]) for _, row in hotpot_data.iterrows()
    ]
    f1_list, correct, results = [], 0, {}
    if max_tasks is not None:
        task_instructions = task_instructions[:max_tasks]
    for test_task, answer in tqdm(task_instructions, desc="Processing"):
        test_task_pack = TaskPackage(instruction=test_task)
        response = agent(test_task_pack)
        execution = agent.short_term_memory.get_action_chain(task=test_task_pack)
        f1, _, _ = f1_score(response, answer)
        f1_list.append(f1)
        correct += int(response == answer)
        results[test_task] = (response, answer)

        avg_f1 = np.mean(f1_list)
        acc = correct / len(task_instructions)
        
        dump_str = f"{test_task}\t{answer}\t{response}\t{f1:.4f}\t{acc:.4f}\t{execution}\n"
        with open(f"data/{agent_arch}_{llm_name}_results_{level}.csv", "a") as f:
            f.write(dump_str)
            
    return avg_f1, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Search Agent on the HotPotQA Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument(
        "--search_mode",
        type=str,
        choices=["online", "rag"],
        default="rag",
        help="Use online Wikipedia API or local RAG retrieval.",
    )
    parser.add_argument(
        "--rag_index_dir",
        type=str,
        default=DEFAULT_RAG_INDEX_DIR,
        help="Directory containing the local Wikipedia vector cache.",
    )
    parser.add_argument(
        "--rag_top_k",
        type=int,
        default=5,
        help="Number of local Wikipedia chunks returned to the LLM per search.",
    )
    parser.add_argument(
        "--rag_strategy",
        type=str,
        default="cosine",
        help="Local retrieval strategy. Currently supports: cosine.",
    )
    parser.add_argument(
        "--rag_embed_model",
        type=str,
        default=DEFAULT_RAG_EMBED_MODEL,
        help="Embedding model used for query encoding.",
    )
    parser.add_argument(
        "--rag_embedding_dim",
        type=int,
        default=DEFAULT_RAG_EMBEDDING_DIM,
        help="Embedding dimension used by the local index.",
    )
    parser.add_argument(
        "--rag_cache_dir",
        type=str,
        default=None,
        help="Optional local cache directory for embedding models.",
    )
    parser.add_argument(
        "--rag_device",
        type=str,
        default=DEFAULT_RAG_DEVICE,
        help="Torch device used for local RAG vectors and query embeddings.",
    )
    parser.add_argument(
        "--rag_text_lookup",
        type=str,
        choices=["cache", "shards"],
        default=DEFAULT_RAG_TEXT_LOOKUP,
        help="Load retrieved text from the full text cache or by scanning shards.",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="If set, only run the first N tasks (smoke test). None = full dataset.",
    )
    args = parser.parse_args()

    f1, acc = run_hotpot_qa_agent(
        level=args.level,
        llm_name=args.llm,
        agent_arch=args.agent_arch,
        PROMPT_DEBUG_FLAG=args.debug,
        search_mode=args.search_mode,
        rag_index_dir=args.rag_index_dir,
        rag_top_k=args.rag_top_k,
        rag_strategy=args.rag_strategy,
        rag_embed_model=args.rag_embed_model,
        rag_embedding_dim=args.rag_embedding_dim,
        rag_cache_dir=args.rag_cache_dir,
        rag_device=args.rag_device,
        rag_text_lookup=args.rag_text_lookup,
        max_tasks=args.max_tasks,
    )
    print(
        f"{'+'*100}\nLLM model: {args.llm}, Dataset: {args.level}, Result: F1-Score = {f1:.4f}, Accuracy = {acc:.4f}"
    )
