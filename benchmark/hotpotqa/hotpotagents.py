from SearchActions import WikipediaSearch

from agentlite.actions import BaseAction, FinishAct, ThinkAct, PlanAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger

class WikiSearchAgent(BaseAgent):
    """
    Agent to search Wikipedia content and answer questions.
    """

    def __init__(self, llm: BaseLLM, reasoning_type: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "wiki_search_agent"
        role = "Answer questions by searching Wikipedia content."
        constraint = "Generation should be simple and clear."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[WikipediaSearch()],
            reasoning_type=reasoning_type,
            constraint=constraint,
            logger=AgentLogger(PROMPT_DEBUG_FLAG=False)
        )
    
    def __build_react_examples__(self):
            
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
        obs1_1 = "OK"

        # 2. First search action and observation
        act_params1 = {"query": "Milhouse"}
        act1_2 = AgentAct(name=WikipediaSearch().action_name, params=act_params1)
        obs1_2 = "Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening. Groening chose the name Milhouse, also the middle name of President Richard Nixon, because it was the most unfortunate name [he] could think of for a kid"

        # 3. Second thought to refine search
        thought1_3 = "Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous."
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
        self.add_example(task=task_pack2, action_chain=act_obs2)
    
    def __build_act_examples__(self):
            
        """
        Constructing the examples for agent working.
        Each example is a successful action-observation chain of an agent.
        These examples should cover all the API calls.
        """

        # Example 1: Question about Milhouse from "The Simpsons"
        task1 = "Musician and satirist Allie Goertz wrote a song about the 'The Simpsons' character Milhouse, who was Matt Groening named after?"

        # 1. First search action and observation
        act_params1 = {"query": "Milhouse"}
        act1_1 = AgentAct(name=WikipediaSearch().action_name, params=act_params1)
        obs1_1 = "Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening. Groening chose the name Milhouse, also the middle name of President Richard Nixon, because it was the most unfortunate name [he] could think of for a kid"

       # 2. Final thought and finish action
        answer1 = "Richard Nixon"
        act1_2 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer1})
        obs1_2 = "Task Completed."

        # Adding example 1 into prompt_gen of the agent
        task_pack1 = TaskPackage(instruction=task1)
        act_obs1 = [
            (act1_1, obs1_1),
            (act1_2, obs1_2)
        ]
        self.add_example(task=task_pack1, action_chain=act_obs1)

        # Example 2: Question about Pavel Urysohn and Leonid Levin
        # Similar steps for constructing the example with search, think, and finish actions
        task2 = "Were Pavel Urysohn and Leonid Levin known for the same type of work?"

        # 1. First search action and observation for Pavel Urysohn
        act_params2_1 = {"query": "Pavel Urysohn"}
        act2_1 = AgentAct(name=WikipediaSearch().action_name, params=act_params2_1)
        obs2_1 = "Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory."

        # 2. Second search action and observation for Leonid Levin
        act_params2_2 = {"query": "Leonid Levin"}
        act2_2 = AgentAct(name=WikipediaSearch().action_name, params=act_params2_2)
        obs2_2 = "Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist."

        # 3. Final thought and finish action for task 2
        answer2 = "yes"
        act2_3 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer2})
        obs2_3 = "Task Completed."

        # Adding example 2 into prompt_gen of the agent
        task_pack2 = TaskPackage(instruction=task2)
        act_obs2 = [
            (act2_1, obs2_1),
            (act2_2, obs2_2),
            (act2_3, obs2_3)
        ]
        self.add_example(task=task_pack2, action_chain=act_obs2)


# test_query = {"query": "Milhouse"}
# test_action = WikipediaSearch()
# a = test_action(**test_query)
# print(a)