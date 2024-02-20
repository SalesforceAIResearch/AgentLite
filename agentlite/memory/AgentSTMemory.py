from typing import Dict, Union

from agentlite.commons import AgentAct, TaskPackage

from .memory_utils import *


class AgentSTMemory:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.memory = None

    def get_action_chain(self, task: TaskPackage):
        raise NotImplementedError

    def add_action(self, action: AgentAct):
        raise NotImplementedError

    def add_new_task(self, task: TaskPackage):
        raise NotImplementedError

    def add_act_obs(self, task: TaskPackage, action: AgentAct, observation: str):
        raise NotImplementedError


class DictAgentSTMemory(AgentSTMemory):
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.memory: Dict[str, Dict[str, Union[TaskPackage, list]]] = {}

    def add_new_task(self, task: TaskPackage):
        self.memory[task.task_id] = {MEMORY_TASK_KEY: task, MEMORY_ACT_OBS_KEY: []}

    def get_action_chain(self, task: TaskPackage):
        return self.memory[task.task_id][MEMORY_ACT_OBS_KEY]

    def add_act_obs(self, task: TaskPackage, action: AgentAct, observation: str = ""):
        """adding action and its corresponding observations into memory"""
        self.memory[task.task_id][MEMORY_ACT_OBS_KEY].append((action, observation))
