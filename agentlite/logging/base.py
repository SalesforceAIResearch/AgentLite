from abc import ABC, abstractmethod

from agentlite.commons import AgentAct, TaskPackage



class BaseAgentLogger(ABC):
    def __init__(
        self,
        log_file_name: str = "agent.log",
    ) -> None:
        self.log_file_name = log_file_name

    def __save_log__(self, log_str: str):
        """how to save the log"""
        pass
        
    def receive_task(self, task: TaskPackage, agent_name: str):
        """the agent receives a task and log it"""
        pass

    def execute_task(self, task: TaskPackage = None, agent_name: str = None, **kwargs):
        """the agent starts to execute the task"""
        pass
    
    def end_execute(self, task: TaskPackage, agent_name: str = None):
        """the agent finishes the task"""
        pass

    def take_action(self, action: AgentAct, agent_name: str, step_idx: int):
        """the agent takes an action"""
        pass

    def get_obs(self, obs: str):
        """get observation"""
        pass

    def get_prompt(self, prompt):
        """get prompt"""
        pass

    def get_llm_output(self, output: str):
        """get llm output"""
        pass