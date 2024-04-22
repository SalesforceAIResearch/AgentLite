from typing import List

from agentlite.actions import FinishAct
from agentlite.agent_prompts import ManagerPromptGen
from agentlite.agent_prompts.prompt_utils import DEFAULT_PROMPT
from agentlite.agents.agent_utils import *
from agentlite.commons import AgentAct, TaskPackage
from agentlite.commons.AgentAct import ActObsChainType
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging import DefaultLogger
from agentlite.logging.terminal_logger import AgentLogger

from .ABCAgent import ABCAgent
from .BaseAgent import BaseAgent


class ManagerAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        name: str = "Base_Manager_Agent",
        role: str = "This is the basic manager agent",
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["manager_instruction"],
        reasoning_type: str = "react",
        TeamAgents: List[ABCAgent] = [],
        logger: AgentLogger = DefaultLogger,
        **kwargs
    ):
        """ManagerAgent inherits BaseAgent. It has all methods for base agent
        and it can communicate with other agent. It controls LaborAgents to complete tasks.
        Also, one can initialize ManagerAgent with a list of PeerAgents
        or add the peerAgent later for discussion.

        :param llm: BaseLLM, the language model for this agent
        :type llm: BaseLLM
        :param name: the name of this agent, defaults to "Base_Manager_Agent"
        :type name: str, optional
        :param role: the role of this agent, defaults to "This is the basic manager agent"
        :type role: str, optional
        :param constraint: the constraints of this agent
        :type constraint: str, optional
        :param instruction: the instruction for this agent
        :type instruction: str, optional
        :param reasoning_type: the reasoning type of this agent, defaults to "react"
        :type reasoning_type: str, optional
        :param TeamAgents: adding a list of agents to this manager agent, defaults to []
        :type TeamAgents: List[ABCAgent], optional
        :param logger: the logger for this agent, defaults to DefaultLogger
        :type logger: AgentLogger, optional
        """
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            constraint=constraint,
            instruction=instruction,
            reasoning_type=reasoning_type,
            logger=logger,
        )
        self.team = TeamAgents
        self.prompt_gen = ManagerPromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )

    def agent_match(self, agent_name: str, agent: ABCAgent) -> bool:
        """math the generated action of agent_name with an agent in the team

        :param agent_name: the agent name
        :type agent_name: str
        :param agent: the agent to match
        :type agent: ABCAgent
        :return: whether the agent_name match the agent
        :rtype: bool
        """
        if agent_name == agent.name:  # exact match
            return True
        # ## To-Do More fuzzy match
        return False

    def add_member(self, LaborAgent: ABCAgent):
        """add a labor agent to the team

        :param LaborAgent: the labor agent to add
        :type LaborAgent: ABCAgent
        """
        self.team.append(LaborAgent)

    def __next_act__(
        self, task: TaskPackage, action_chain: ActObsChainType
    ) -> AgentAct:
        """one-step action generation for manager agent

        :param task: the next action towards the task
        :type task: TaskPackage
        :param action_chain: history actions and observation of this task from memory
        :type action_chain: ActObsChainType
        :return: action for agent to execute
        :rtype: AgentAct
        """
        labor_agents_doc = {
            labor_agent.name: labor_agent.role for labor_agent in self.team
        }
        action_prompt = self.prompt_gen.action_prompt(
            task=task,
            actions=self.actions,
            action_chain=action_chain,
            labor_agents_doc=labor_agents_doc,
        )
        self.logger.get_prompt(action_prompt)
        raw_action = self.llm_layer(action_prompt)
        self.logger.get_llm_output(raw_action)
        return self.__action_parser__(raw_action)

    def __action_parser__(self, raw_action: str) -> AgentAct:
        """parse the raw action from llm to AgentAct

        :param raw_action: the raw action from llm
        :type raw_action: str
        :return: the parsed action
        :rtype: AgentAct
        """

        action_name, args, PARSE_FLAG = parse_action(raw_action)
        # if action_name match a labor_agent
        if self.team:
            for agent in self.team:
                if self.agent_match(action_name, agent):
                    agent_act = AgentAct(name=action_name, params=args)

        # if action_name is action
        for action in self.actions:
            if act_match(action_name, action):
                agent_act = AgentAct(name=action_name, params=args)
        return agent_act

    def forward(self, task: TaskPackage, agent_act: AgentAct) -> str:
        """forward the action to get the observation or response from other agent

        :param task: the task to forward
        :type task: TaskPackage
        :param agent_act: the action to forward
        :type agent_act: AgentAct
        :return: the observation or response from other agent
        :rtype: str
        """
        act_found_flag = False
        # if action is labor agent call
        for agent in self.team:
            if self.agent_match(agent_act.name, agent):
                act_found_flag = True
                new_task_package = self.create_TP(
                    agent_act.params[AGENT_CALL_ARG_KEY], agent.id
                )
                observation = agent(new_task_package)
                return observation
        # if action is inner action
        if agent_act.name == FinishAct.action_name:
            act_found_flag = True
            observation = "Task Completed."
            task.completion = "completed"
            task.answer = FinishAct(**agent_act.params)
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

    def create_TP(self, task_ins: str, executor: str) -> TaskPackage:
        """create a task package for labor agent

        :param task_ins: the instruction of the task
        :type task_ins: str
        :param executor: the executor name of the task, an agent name
        :type executor: str
        :return: the task package
        :rtype: TaskPackage
        """
        task = TaskPackage(
            instruction=task_ins, task_creator=self.id, task_executor=executor
        )
        return task
