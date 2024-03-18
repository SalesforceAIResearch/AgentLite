from typing import List

from agentlite.actions import BaseAction, FinishAct, ThinkAct, PlanAct
from agentlite.agent_prompts import BasePromptGen
from agentlite.agent_prompts.prompt_utils import DEFAULT_PROMPT
from agentlite.agents.agent_utils import *
from agentlite.commons import AgentAct, TaskPackage
from agentlite.commons.AgentAct import ActObsChainType
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging import DefaultLogger
from agentlite.logging.multi_agent_log import AgentLogger
from agentlite.memory.AgentSTMemory import AgentSTMemory, DictAgentSTMemory

from .ABCAgent import ABCAgent


class BaseAgent(ABCAgent):
    """the base agent class for multi-turn action calling. Subclass from ABCAgent

    :param name: the name of this agent
    :type name: str
    :param role: the role of this agent
    :type role: str
    :param llm: the language model for this agent
    :type llm: BaseLLM
    :param actions: the action space that the agent can choose from, defaults to []
    :type actions: List[BaseAction], optional
    :param constraint: the constraints of this agent , defaults to "You generation should be simple and clear."
    :type constraint: str, optional
    :param instruction: the agent instruction, defaults to "You are an intelligent agent.\
        You should follow your {PROMPT_TOKENS["role"]['begin']}, {PROMPT_TOKENS["action"]['begin']} to take actions.\
            Your generation should follow the example format. Finish the task as best as you can.". 
            PROMPT_TOKENS is defined in agentlite/agent_prompts/prompt_utils.py
    :type instruction: str, optional
    :param reasoning_type: the reasoning type of this agent, defaults to "react"
    :type reasoning_type: str, optional
    :param logger: the logger for this agent, defaults to DefaultLogger
    :type logger: AgentLogger, optional

    Methods:
        - __call__(task: TaskPackage) -> str
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm: BaseLLM,
        actions: List[BaseAction] = [],
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        reasoning_type: str = "react",
        logger: AgentLogger = DefaultLogger,
        **kwargs
    ):
        super().__init__(name=name, role=role)
        self.name = name  # short description for agent, use it for id part
        self.role = role  # describe the job duty of this agent
        self.llm = llm
        self.actions = actions
        self.max_exec_steps = 20
        self.task_pool = []
        self.constraint = constraint
        self.instruction = instruction
        self.reasoning_type = reasoning_type
        self.prompt_gen = BasePromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )
        self.logger = logger
        self.__add_st_memory__()
        self.__add_inner_actions__()

    def __add_st_memory__(self, short_term_memory: AgentSTMemory = None):
        """adding short-term memory to agent

        :param short_term_memory: the short-term memory, defaults to None
        :type short_term_memory: AgentSTMemory, optional
        """ """
        """
        if short_term_memory:
            self.short_term_memory = short_term_memory
        else:
            self.short_term_memory = DictAgentSTMemory(agent_id=self.id)

    def __add_inner_actions__(self):
        """adding the inner action types into agent, which is based on the `self.reasoning_type`"""
        if self.reasoning_type == "react":
            self.actions += [ThinkAct, FinishAct]
        elif self.reasoning_type == "act":
            self.actions += [FinishAct]
        elif self.reasoning_type == "planact":
            self.actions += [PlanAct, FinishAct]
        elif self.reasoning_type == "planreact":
            self.actions += [PlanAct, ThinkAct, FinishAct]
        else:
            Warning("Not yet supported. Will using react instead.")
            self.actions += [ThinkAct, FinishAct]
        self.actions = list(set(self.actions))

    def __call__(self, task: TaskPackage) -> str:
        """agent can be called with a task. it will assign the task and then execute and respond

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :return: the response of this task
        :rtype: str
        """
        # adding log information
        self.logger.receive_task(task=task, agent_name=self.name)
        self.assign(task)
        self.execute(task)
        response = self.respond(task)
        return response

    def assign(self, task: TaskPackage) -> None:
        """assign task to agent

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        self.short_term_memory.add_new_task(task)
        self.task_pool.append(task)

    def llm_layer(self, prompt: str) -> str:
        """input a prompt, llm generates a text

        :param prompt: the prompt string
        :type prompt: str
        :return: the output from llm, which is a string
        :rtype: str
        """
        return self.llm.run(prompt)

    def execute(self, task: TaskPackage):
        """multi-step execution of actions. Generate the actions for a task until reach the done

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        step_size = 0
        self.logger.execute_task(task=task, agent_name=self.name)
        while task.completion == "active" and step_size < self.max_exec_steps:
            action_chain = self.short_term_memory.get_action_chain(task)
            action = self.__next_act__(task, action_chain)
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size)
            observation = self.forward(task, action)
            self.logger.get_obs(obs=observation)
            self.__st_memorize__(task, action, observation)
            step_size += 1
        self.logger.end_execute(task=task, agent_name=self.name)

    def respond(self, task: TaskPackage, **kwargs) -> str:
        """generate messages for manager agents

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :return: a response
        :rtype: str
        """

        if task.completion in ["completed"]:
            return task.answer
        else:
            # to do: add more actions after the task is not completed such as summarizing the actions
            return DEFAULT_PROMPT["not_completed"]

    def __next_act__(
        self, task: TaskPackage, action_chain: ActObsChainType
    ) -> AgentAct:
        """one-step action generation

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param action_chain: history actions and observation of this task from memory
        :type action_chain: ActObsChainType
        :return: the action for agent to execute
        :rtype: AgentAct
        """

        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=action_chain,
        )
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)
        return self.__action_parser__(raw_action)

    def __st_memorize__(
        self, task: TaskPackage, action: AgentAct, observation: str = ""
    ):
        """the short-term memorize action and observation for agent

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param action: the action wrapper for execution
        :type action: AgentAct
        :param observation: the observation after action execution, defaults to ""
        :type observation: str, optional
        """
        self.short_term_memory.add_act_obs(task, action, observation)

    def __action_parser__(self, raw_action: str) -> AgentAct:
        """parse the generated content to an executable action

        :param raw_action: llm generated text
        :type raw_action: str
        :return: an executable action wrapper
        :rtype: AgentAct
        """

        action_name, args, PARSE_FLAG = parse_action(raw_action)
        agent_act = AgentAct(name=action_name, params=args)
        return agent_act

    def forward(self, task: TaskPackage, agent_act: AgentAct) -> str:
        """
        using this function to forward the action to get the observation.

        :param task: the task which agent receives and solves.
        :type task: TaskPackage
        :param agent_act: the action wrapper for execution.
        :type agent_act: AgentAct
        :return: observation
        :rtype: str
        """
        act_found_flag = False
        # if action is Finish Action
        if agent_act.name == FinishAct.action_name:
            act_found_flag = True
            observation = "Task Completed."
            task.completion = "completed"
            task.answer = FinishAct(**agent_act.params)
        # if match one in self.actions
        else:
            for action in self.actions:
                if act_match(agent_act.name, action):
                    act_found_flag = True
                    observation = action(**agent_act.params)
        # if not find this action
        if act_found_flag:
            return observation
        else:
            observation = ACION_NOT_FOUND_MESS
        return observation

    def add_example(
        self,
        task: TaskPackage,
        action_chain: List[tuple[AgentAct, str]],
        example_type: str = "action",
    ):
        """add an example to prompt generator

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param action_chain: the action chain of this task
        :type action_chain: List[tuple[AgentAct, str]]
        :param example_type: the type of this example, defaults to "action"
        :type example_type: str, optional
        """
        self.prompt_gen.add_example(task, action_chain, example_type=example_type)
