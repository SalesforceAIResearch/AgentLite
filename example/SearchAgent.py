from typing import List

from SearchActions import DuckSearch, WikipediaSearch

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger

# set PROMPT_DEBUG_FLAG to True to see the debug info
agent_logger = AgentLogger(PROMPT_DEBUG_FLAG=False)


class SearchAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        role: str,
        llm: BaseLLM,
        actions: List[BaseAction] = [WikipediaSearch()],
        manager: ABCAgent = None,
        **kwargs
    ):
        super().__init__(
            name=name, role=role, llm=llm, actions=actions, manager=manager, **kwargs
        )


class WikiSearchAgent(BaseAgent):
    def __init__(
        self, llm: BaseLLM, actions: List[BaseAction] = [WikipediaSearch()], **kwargs
    ):
        name = "wiki_search_agent"
        role = "You can answer questions by searching wikipedia content."
        super().__init__(
            name=name, role=role, llm=llm, actions=actions, logger=agent_logger
        )
        self.__build_examples__()

    def __build_examples__(self):
        """
        constructing the examples for agent working.
        Each example is a successful action-obs chain of an agent.
        those examples should cover all those api calls
        """
        # an example of search agent with wikipedia api call
        # task
        task = "what is the found date of salesforce?"

        # 1. think action and obs
        thought = "I should first use Wikipedia_Search to search salesforce"
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = ""

        # 2. api call action and obs
        act_params = {"query": "salesforce"}
        act_2 = AgentAct(name=WikipediaSearch().action_name, params=act_params)
        obs_2 = """Page: Salesforce
        Summary: Salesforce, Inc. is an American cloud-based software company headquartered in San Francisco, California. It provides customer relationship management (CRM) software and applications focused on sales, customer service, marketing automation, e-commerce, analytics, and application development.
        Founded by former Oracle executive Marc Benioff in February 1999, Salesforce grew quickly, making its IPO in 2004. As of September 2022, Salesforce is the 61st largest company in the world by market cap with a value of nearly US$153 billion. Salesforce's rapid growth made it the first cloud computing company to reach US$1 billion in annual revenue, which it achieved in fiscal year 2009. It became the world's largest enterprise software firm in 2022. Salesforce ranked 136th on the most recent edition of the Fortune 500, making US$26.5 billion in 2022. Since 2020, Salesforce has also been a component of the Dow Jones Industrial Average."""

        # 3. think action and obs
        thought = "I find salesforce is Founded by former Oracle executive Marc Benioff in February 1999"
        act_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_3 = "OK"

        # 4. finish action
        answer = "February 1999"
        act_4 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_4 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
        self.add_example(task=task_pack, action_chain=act_obs)


class DuckSearchAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [DuckSearch()],
        manager: ABCAgent = None,
        **kwargs
    ):
        name = "duck_search_agent"
        role = "You can answer questions by using duck duck go search content."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            manager=manager,
            logger=agent_logger,
        )


def test_search_agent():
    llm_config_dict = {"llm_name": "gpt-3.5-turbo-16k-0613", "temperature": 0.1}
    actions = [WikipediaSearch()]
    llm_config = LLMConfig(llm_config_dict)
    # print(llm_config.__dict__)
    llm = get_llm_backend(llm_config)
    ## test the one-shot wikipedia search agent
    labor_agent = WikiSearchAgent(llm=llm)
    # labor_agent = DuckSearchAgent(llm=llm)

    test_task = "what is the found date of microsoft"
    test_task_pack = TaskPackage(instruction=test_task)
    response = labor_agent(test_task_pack)
    print("response:", response)


if __name__ == "__main__":
    test_search_agent()
