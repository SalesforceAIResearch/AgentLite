from typing import List
from SearchActions import WikipediaSearch

from agentlite.actions import BaseAction, FinishAct, ThinkAct, PlanAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agent_prompts.prompt_utils import DEFAULT_PROMPT
from agentlite.agents.BaseAgent import BaseAgent
from agentlite.agents.ManagerAgent import ManagerAgent
from agentlite.agents.agent_utils import AGENT_CALL_ARG_KEY
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger



class SearchAgent(BaseAgent):
    """
    Agent to search Wikipedia content and answer questions.
    """

    def __init__(self, llm: BaseLLM, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "search_agent"
        role = "Answer questions by searching Wikipedia content."
        constraint = "Your response should be simple and clear."
        instruction = "You are an intelligent search_agent. You should follow your [Role], [Action_Doc] to take actions. You should decompose your task into executable actions. Your generation should follow the example format. Finish the task if you find the answer. And you answer should be simple and straighforward. DO NOT repeat your actions."
        self.agent_arch = agent_arch
        if agent_arch in ["zs"]:
            reasoning_type = "act"
        elif agent_arch in ["zst"]:
            reasoning_type = "react"
        else:
            reasoning_type = agent_arch

        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[WikipediaSearch()],
            reasoning_type=reasoning_type,
            constraint=constraint,
            instruction=instruction, # common instruction will use default in agentlite.agent_prompts.prompt_utils.DEFAULT_PROMPT["agent_instruction"]
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        # self.__build_examples__()

    
class ReasonAgent(ManagerAgent):
    def __init__(self, llm: BaseLLM, search_agent: BaseAgent, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False ):
        name = "reasoning_agent"
        role = "You are an intelligent reasoning agent that can setup a reasoning steps and ask a search_agent to search for the answer."
        constraint = "Your answer should be simple and clear."
        instruction = "You are an intelligent Wikipedia agent. You should follow your [Role], [Action_Doc], [Team_Doc] to take actions. You should decompose the questions into multi-hop reasoning steps. Your generation should follow the example format. You should assign the search task to the search_agent if you need more online information. Finish the task with a simple and direct answer."
        super().__init__(
            name=name,
            role=role,
            constraint=constraint,
            llm=llm,
            TeamAgents=[search_agent],
            reasoning_type=agent_arch,
            instruction=instruction, # common instruction will use default in agentlite.agent_prompts.prompt_utils.DEFAULT_PROMPT["agent_instruction"]
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.agent_arch = agent_arch
        self.search_agent = search_agent
        self.__build_examples__()
    
    def __build_examples__(self):
        if self.agent_arch == "react":
            self.__build_react_examples__()
        
    def __build_react_examples__(self):
            
        """
        Constructing the examples for agent working.
        Each example is a successful action-observation chain of an agent.
        These examples should cover all the API calls.
        """

        # Example 1: Question about Milhouse from "The Simpsons"
        task1 = "Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' character Milhouse, who was Matt Groening named after?"

        # 1. Initial thought and default observation
        thought1_1 = "The question simplifies to 'The Simpsons' character Milhouse is named after who. I only need to search Milhouse and find who it is named after."
        act1_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought1_1})
        obs1_1 = "OK"

        # 2. First ask search agent to search for the simpsons character Milhouse
        act_params1 = {AGENT_CALL_ARG_KEY: "findout who Milhouse from The Simpsons is named after."}
        act1_2 = AgentAct(name=self.search_agent.name, params=act_params1)
        obs1_2 = "Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous."

        # 3. Second thought to refine search
        thought1_3 = "Now I know Milhouse was named after U.S. president Richard Nixon"
        act1_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought1_3})
        obs1_3 = "OK"

       # 5. Final thought and finish action
        answer1 = "Richard Nixon"
        act1_4 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer1})
        obs1_4 = "Task Completed."

        # Adding example 1 into prompt_gen of the agent
        task_pack1 = TaskPackage(instruction=task1)
        act_obs1 = [
            (act1_1, obs1_1),
            (act1_2, obs1_2),
            (act1_3, obs1_3),
            (act1_4, obs1_4)
        ]
        self.add_example(task=task_pack1, action_chain=act_obs1)

        # Example 2: Question about Pavel Urysohn and Leonid Levin
        # Similar steps for constructing the example with search, think, and finish actions
        task2 = "Were Pavel Urysohn and Leonid Levin known for the same type of work?"

        # 1. Initial thought and empty observation for task 2
        thought2_1 = "I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same."
        act2_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought2_1})
        obs2_1 = "OK"

        # 2. First search action and observation for Pavel Urysohn
        act_params2_1 = {AGENT_CALL_ARG_KEY: "what is the work of Pavel Urysohn?"}
        act2_2 = AgentAct(name=self.search_agent.name, params=act_params2_1)
        obs2_2 = "Pavel Samuilovich Urysohn was a Soviet mathematician who is best known for his contributions in dimension theory."

        # 3. Second thought for searching Leonid Levin
        thought2_3 = "Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work."
        act2_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought2_3})
        obs2_3 = "OK"

        # 4. Second search action and observation for Leonid Levin
        act_params2_2 = {AGENT_CALL_ARG_KEY: "what is the work of Leonid Levin?"}
        act2_4 = AgentAct(name=self.search_agent.name, params=act_params2_2)
        obs2_4 = "Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist."

        # 5. Final thought and finish action for task 2
        thought2_5 = "Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work."
        act2_5 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought2_5})
        obs2_5 = "OK"
        
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
        self.add_example(task=task_pack2, action_chain=act_obs2)

if __name__ == "__main__":
    # Load the GPT-3.5-turbo-16k-0613 language model
    config_dict = {
        "llm_name": "gpt-4",
        "temperature": 0.0,
    }
    llm_config = LLMConfig(config_dict=config_dict)
    llm = get_llm_backend(llm_config)

    # Create the search agent
    search_agent = SearchAgent(llm=llm, agent_arch="react", PROMPT_DEBUG_FLAG=False)

    # Create the reasoning agent
    reasoning_agent = ReasonAgent(llm=llm, search_agent=search_agent, agent_arch="react", PROMPT_DEBUG_FLAG=False)
    reasoning_agent.__build_examples__()

    # Test the reasoning agent on a question about Milhouse from "The Simpsons"
    task = "Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' character Milhouse, who was Matt Groening named after?"
    task_pack = TaskPackage(instruction=task)
    reasoning_agent(task_pack)

    # # Test the reasoning agent on a question about Pavel Urysohn and Leonid Levin
    # task = "Were Pavel Urysohn and Leonid Levin known for the same type of work?"
    # task_pack = TaskPackage(instruction=task)
    # reasoning_agent(task_pack)