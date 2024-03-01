from webshop_actions import ClickAction, SearchAction

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger

agent_logger = AgentLogger(PROMPT_DEBUG_FLAG=False)


def build_act_agent(
    session_idx, llm_config_dict: dict, logger: AgentLogger = agent_logger
):
    llm_config = LLMConfig(llm_config_dict)
    llm = get_llm_backend(llm_config)
    webshop_agent = BaseAgent(
        name="webshop_agent",
        role="You can interact with the webshop",
        llm=llm,
        actions=[
            ClickAction(session_idx=session_idx),
            SearchAction(session_idx=session_idx),
        ],
        reasoning_type="act",
        logger=logger,
    )

    # build an example
    search_act = SearchAction(session_idx)
    click_act = ClickAction(session_idx)
    example_task = "i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars"
    exp_task_package = TaskPackage(instruction=example_task)
    act_1 = AgentAct(
        name=search_act.action_name,
        params={"product": "bright citrus deodorant for sensitive skin"},
    )
    obs_1 = """WEB PAGE: {
    [Back to Search]
    Page 1 (Total results: 15)
    [Next >]
    [B078GWRC1J]
    Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
    $10.99
    [B08KBVJ4XN]
    Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)
    $15.95
    [B08KC3QXZW]
    Barrel and Oak - Aluminum-Free Deodorant Variety Pack, Deodorant for Men, Natural Fragrance, 3 Essential Oil-Based Scents, For Sensitive Skin, No Clothing Stains, Vegan (3 oz Per Deodorant, 3-Pack)
    $26.95}"""
    act_2 = AgentAct(name=click_act.action_name, params={"button": "B078GWRC1J"})
    obs_2 = """WEB PAGE: {
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [3 ounce (pack of 1)]}"""
    act_3 = AgentAct(name=click_act.action_name, params={"button": "bright citrus"})
    obs_3 = """"You have clicked bright citrus."""
    act_4 = AgentAct(
        name=click_act.action_name, params={"button": "3 ounce (pack of 1)"}
    )
    obs_4 = """"You have clicked 3 ounce (pack of 1)."""
    act_5 = AgentAct(name=click_act.action_name, params={"button": "Buy Now"})
    obs_5 = """"You have bought the product."""
    act_6 = AgentAct(
        name=FinishAct.action_name, params={INNER_ACT_KEY: "Task Finished. Reward: 1.0"}
    )
    obs_6 = """"You have finished the shopping. Reward: 1.0"""
    webshop_agent.add_example(
        task=exp_task_package,
        action_chain=[
            (act_1, obs_1),
            (act_2, obs_2),
            (act_3, obs_3),
            (act_4, obs_4),
            (act_5, obs_5),
            (act_6, obs_6),
        ],
    )
    return webshop_agent
