import os
import chainlit as cl
import streamlit as st


from agentlite.commons import AgentAct, TaskPackage
from agentlite.logging.utils import *

from .base import BaseAgentLogger

class UILogger(BaseAgentLogger):
    def __init__(
        self,
        log_file_name: str = "agent.log",
        FLAG_PRINT: bool = True,
        OBS_OFFSET: int = 500,
        PROMPT_DEBUG_FLAG: bool = False,
    ) -> None:
        self.log_file_name = log_file_name
        self.FLAG_PRINT = FLAG_PRINT  # whether print the log into terminal
        self.OBS_OFFSET = OBS_OFFSET
        self.PROMPT_DEBUG_FLAG = PROMPT_DEBUG_FLAG
        
    def __save_log__(self, log_str: str):
        with st.chat_message("assistant"):
            st.markdown(log_str)
        st.session_state.messages.append({"role": "assistant", "content": log_str})
            
    def receive_task(self, task: TaskPackage, agent_name: str):
        log_str = f"""Agent {agent_name} """
        log_str += f"""receives the following TaskPackage:\n"""
        
    def execute_task(self, task: TaskPackage = None, agent_name: str = None, **kwargs):
        log_str = f"""{agent_name} starts execution on TaskPackage {task.task_id}===="""
        # self.__save_log__(log_str=log_str)

    def end_execute(self, task: TaskPackage, agent_name: str = None):
        log_str = f"""{agent_name} finish execution. TaskPackage[ID:{task.task_id}] status:\n"""
        task_str = f"""[\n\tcompletion: {task.completion}\n\tanswer: {task.answer}\n]"""
        # log_str += self.__color_task_str__(task_str=task_str)
        # log_str += "\n=========="
        # self.__save_log__(log_str=log_str)

    def take_action(self, action: AgentAct, agent_name: str, step_idx: int):
        act_str = f"""{{\n\tname: {action.name}\n\tparams: {action.params}\n}}"""
        log_str = f"""**{agent_name}** takes **{step_idx}-step** Action:\n"""
        log_str += f"""```json
        {act_str}```"""
        self.__save_log__(log_str)

    def get_obs(self, obs: str):
        if len(obs) > self.OBS_OFFSET:
            obs = obs[: self.OBS_OFFSET] + "[TLDR]"
        log_str = f"""**Observation:** ```{obs}```"""
        self.__save_log__(log_str)

    def get_prompt(self, prompt):
        log_str = f"""Prompt: {prompt}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)

    def get_llm_output(self, output: str):
        log_str = f"""LLM generates: {output}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)