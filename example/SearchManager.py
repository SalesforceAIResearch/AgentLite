from typing import List, Optional

from SearchAgent import DuckSearchAgent, WikiSearchAgent

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent, ManagerAgent
from agentlite.agents.agent_utils import AGENT_CALL_ARG_KEY
from agentlite.commons import ActObsChainType, AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger

# set PROMPT_DEBUG_FLAG to True to see the debug info
agent_logger = AgentLogger(PROMPT_DEBUG_FLAG=False)


class SearchManager(ManagerAgent):
    def __init__(self, llm: BaseLLM, TeamAgents: List[ABCAgent] = None, **kwargs):
        super().__init__(
            llm,
            name="Search_Manager",
            role="Controlling multiple search agents to complete a search task.",
            TeamAgents=TeamAgents,
            logger=agent_logger,
        )


def test_manager_agent():
    # setting the llm config of manager agent
    llm_config_dict = {
        "llm_name": "gpt-3.5-turbo",
        "temperature": 0.9,
        "context_len": 4000,
    }
    llm_config = LLMConfig(llm_config_dict)
    llm = get_llm_backend(llm_config)

    # setting the team of manager agent
    wiki_search_agent = WikiSearchAgent(llm)
    duck_search_agent = DuckSearchAgent(llm)
    team = [wiki_search_agent, duck_search_agent]

    # initialize the manager with llm and team labor agent
    search_manager = SearchManager(llm, TeamAgents=team)

    # adding one example to manager agent
    exp_task = "who is the founder of salesforce"
    exp_task_pack = TaskPackage(instruction=exp_task)
    act_1 = AgentAct(
        name=ThinkAct.action_name,
        params={
            INNER_ACT_KEY: f"I should first ask {duck_search_agent.name} to search salesforce, if I didn't find an answer, I will ask {wiki_search_agent.name} to search salesforce"
        },
    )
    obs_1 = "OK"
    act_2 = AgentAct(
        name=duck_search_agent.name,
        params={AGENT_CALL_ARG_KEY: "search the information of salesforce"},
    )
    obs_2 = """Page: Salesforce
    Summary: Salesforce, Inc. is an American cloud-based software company headquartered in San Francisco, California. It provides customer relationship management (CRM) software and applications focused on sales, customer service, marketing automation, e-commerce, analytics, and application development.
    Founded by former Oracle executive Marc Benioff in February 1999, Salesforce grew quickly, making its IPO in 2004. As of September 2022, Salesforce is the 61st largest company in the world by market cap with a value of nearly US$153 billion. Salesforce's rapid growth made it the first cloud computing company to reach US$1 billion in annual revenue, which it achieved in fiscal year 2009. It became the world's largest enterprise software firm in 2022. Salesforce ranked 136th on the most recent edition of the Fortune 500, making US$26.5 billion in 2022. Since 2020, Salesforce has also been a component of the Dow Jones Industrial Average."""
    act_3 = AgentAct(
        name=ThinkAct.action_name,
        params={INNER_ACT_KEY: "I find salesforce is founded by Marc Benioff"},
    )
    obs_3 = ""
    act_4 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Marc Benioff"})
    obs_4 = "Task Completed."
    exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
    search_manager.add_example(task=exp_task_pack, action_chain=exp_act_obs)

    # run test
    test_task = "what is micorsoft famous for"
    test_task_pack = TaskPackage(instruction=test_task, task_creator="User")
    response = search_manager(test_task_pack)
    print(response)


if __name__ == "__main__":
    test_manager_agent()
