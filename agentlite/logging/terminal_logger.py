import os

from agentlite.commons import AgentAct, TaskPackage
from agentlite.logging.utils import *
from agentlite.utils import bcolors

from .base import BaseAgentLogger

class AgentLogger(BaseAgentLogger):
    def __init__(
        self,
        log_file_name: str = "agent.log",
        FLAG_PRINT: bool = True,
        OBS_OFFSET: int = 100,
        PROMPT_DEBUG_FLAG: bool = False,
    ) -> None:
        super().__init__(log_file_name=log_file_name)
        self.FLAG_PRINT = FLAG_PRINT  # whether print the log into terminal
        self.OBS_OFFSET = OBS_OFFSET
        self.PROMPT_DEBUG_FLAG = PROMPT_DEBUG_FLAG

    def __color_agent_name__(self, agent_name: str):
        return f"""{bcolors.OKBLUE}{agent_name}{bcolors.ENDC}"""

    def __color_task_str__(self, task_str: str):
        return f"""{bcolors.OKCYAN}{task_str}{bcolors.ENDC}"""

    def __color_act_str__(self, act_str: str):
        return f"""{bcolors.OKBLUE}{act_str}{bcolors.ENDC}"""

    def __color_obs_str__(self, act_str: str):
        return f"""{bcolors.OKGREEN}{act_str}{bcolors.ENDC}"""

    def __color_prompt_str__(self, prompt: str):
        return f"""{bcolors.WARNING}{prompt}{bcolors.ENDC}"""

    def __save_log__(self, log_str: str):
        if self.FLAG_PRINT:
            print(log_str)
        with open(self.log_file_name, "a") as f:
            f.write(str_color_remove(log_str) + "\n")

    def receive_task(self, task: TaskPackage, agent_name: str):
        task_str = (
            f"""[\n\tTask ID: {task.task_id}\n\tInstruction: {task.instruction}\n]"""
        )
        log_str = f"""Agent {self.__color_agent_name__(agent_name)} """
        log_str += f"""receives the following {bcolors.UNDERLINE}TaskPackage{bcolors.ENDC}:\n"""
        log_str += f"{self.__color_task_str__(task_str=task_str)}"
        self.__save_log__(log_str=log_str)

    def execute_task(self, task: TaskPackage = None, agent_name: str = None, **kwargs):
        log_str = f"""===={self.__color_agent_name__(agent_name)} starts execution on TaskPackage {task.task_id}===="""
        self.__save_log__(log_str=log_str)

    def end_execute(self, task: TaskPackage, agent_name: str = None):
        log_str = f"""========={self.__color_agent_name__(agent_name)} finish execution. TaskPackage[ID:{task.task_id}] status:\n"""
        task_str = f"""[\n\tcompletion: {task.completion}\n\tanswer: {task.answer}\n]"""
        log_str += self.__color_task_str__(task_str=task_str)
        log_str += "\n=========="
        self.__save_log__(log_str=log_str)

    def take_action(self, action: AgentAct, agent_name: str, step_idx: int):
        act_str = f"""{{\n\tname: {action.name}\n\tparams: {action.params}\n}}"""
        log_str = f"""Agent {self.__color_agent_name__(agent_name)} takes {step_idx}-step {bcolors.UNDERLINE}Action{bcolors.ENDC}:\n"""
        log_str += f"""{self.__color_act_str__(act_str)}"""
        self.__save_log__(log_str)

    def add_st_memory(self, agent_name: str):
        log_str = f"""Action and Observation added to Agent {self.__color_agent_name__(agent_name)} memory"""
        self.__save_log__(log_str)

    def get_obs(self, obs: str):
        if len(obs) > self.OBS_OFFSET:
            obs = obs[: self.OBS_OFFSET] + "[TLDR]"
        log_str = f"""Observation: {self.__color_obs_str__(obs)}"""
        self.__save_log__(log_str)

    def get_prompt(self, prompt):
        log_str = f"""Prompt: {self.__color_prompt_str__(prompt)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)

    def get_llm_output(self, output: str):
        log_str = f"""LLM generates: {self.__color_prompt_str__(output)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)