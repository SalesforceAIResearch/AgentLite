from webshop_actions import ClickAction, SearchAction

from agentlite.actions import BaseAction, FinishAct, ThinkAct, PlanAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger

class SearchAgent(BaseAgent):
    """
    webshop search agent
    """
    def __init__(self, session_idx, llm: BaseLLM, agent_arch: str = "act", PROMPT_DEBUG_FLAG=False):
        name = "search_agent"
        role = "You can search items on the webshop."
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
            actions=[
                SearchAction(session_idx=session_idx),
            ],
            reasoning_type=reasoning_type,
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.__build_examples__()
    
    def __build_examples__(self):
        # build an example
        search_act = SearchAction("fixed_999")
        example_task_1 = "i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars"
        exp_task_package = TaskPackage(instruction=example_task_1)
        act_1_1 = AgentAct(
            name=search_act.action_name,
            params={"product": "bright citrus deodorant for sensitive skin"},
        )
        obs_1_1 = """WEB PAGE: {
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
        act_1_2 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Search is finished."})
        obs_1_2 = "Search Completed"
        self.add_example(
            task=exp_task_package,
            action_chain=[
                (act_1_1, obs_1_1),
                (act_1_2, obs_1_2)
            ],
        )
        example_task_2 = """i'm looking for a yellow hair styling product that is made from natural ingredients and easy to use, and price lower than 40.00 dollars
You have already searched [yellow hair styling product made from natural ingredients easy to use price < 40.00 dollars]. Could you please change your search query?"""
        act_2_1 = AgentAct(
            name=search_act.action_name,
            params={"product": "yellow hair styling natural ingredients"},
        )
        obs_2_1 = """WEB PAGE: {
[Back to Search] 
Page 1 (Total results: 15) 
[Next >] 
[B09SHK8137] 
Revitalizing Shampoo, Purple Shampoo Remove Yellow, Shampoo To Remove Yellow From Gray Hair, Natural Purple Shampoo And Conditioner Set, Protective Uv Filter Moisturizing Hair Care Lotion (100ml) 
$13.88 
[B08668NQJ1] 
FOXTAIL Violet Velvet Toning Hair Mask - Enhance Blonde & Grey Highlights and Hair Colors - Gentle Light Non-Staining Toning Perfect for Dry, Delicate and Over Processed Bleached Hair - Antioxidant Enriched & Paraben Free - 4 Fl Oz 
$36.0 
[B08LYZJCX1] 
OGX Blonde Enhance + Purple Toning Drops, Blonde Toning to Personalize Your Blonde, Silver, Pre-lightened, Natural Blonde Hair, with Keratin, Sulfate Surfactant Free, Paraben Free, 4oz 
$7.47 }"""
        act_2_2 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Search is finished."})
        obs_2_2 = "Search Completed"
        self.add_example(
            task=TaskPackage(instruction=example_task_2),
            action_chain=[
                (act_2_1, obs_2_1),
                (act_2_2, obs_2_2)
            ],
        )
        
class ClickAgent(BaseAgent):
    """
    webshop search agent
    """
    def __init__(self, session_idx, llm: BaseLLM, agent_arch: str = "act", PROMPT_DEBUG_FLAG=False):
        name = "click_agent"
        role = "You can click buttons on the webshop."
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
            actions=[
                ClickAction(session_idx=session_idx),
            ],
            reasoning_type=reasoning_type,
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.__build_examples__()
    
    def __build_examples__(self):
        # build an example
        click_act = ClickAction("fixed_999")
        example_task_1 = """i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars. \nWEB PAGE: {
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
        exp_task_package = TaskPackage(instruction=example_task_1)
        act_1_1 = AgentAct(
            name=click_act.action_name,
            params={"button": "B078GWRC1J"},
        )
        obs_1_1 = """WEB PAGE: {
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Attributes] 
[Buy Now] }"""
        act_1_2 = AgentAct(name=click_act.action_name, params={"button": "bright citrus"})
        obs_1_2 = "WEB PAGE: {You have clicked bright citrus.}"
        act_1_3 = AgentAct(name=click_act.action_name, params={"button": "3 ounce (pack of 1)"})
        obs_1_3 = "WEB PAGE: {You have clicked 3 ounce (pack of 1).}"
        act_1_4 = AgentAct(name=click_act.action_name, params={"button": "Buy Now"})
        obs_1_4 = "WEB PAGE: {Result: [Success]}"
        act_1_5 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Finish shoppping"})
        obs_1_5 = "Finish shoppping"
        self.add_example(
            task=exp_task_package,
            action_chain=[
                (act_1_1, obs_1_1),
                (act_1_2, obs_1_2),
                (act_1_3, obs_1_3),
                (act_1_4, obs_1_4),
                (act_1_5, obs_1_5)
            ],
        )

class bolaa_webagent:
    def __init__(self, session_idx, env, llm: BaseLLM, PROMPT_DEBUG_FLAG=False):
        self.webshop_env = env
        self.goal: str = ""
        self.observation: str = ""
        self.search_query_list: list = None
        self._reset(session_idx)
        self.search_agent = SearchAgent(session_idx, llm, PROMPT_DEBUG_FLAG)
        self.click_agent = ClickAgent(session_idx, llm, PROMPT_DEBUG_FLAG)
        
    def _reset(self, session_idx):
        action = "reset[]"
        self.observation, self.reward, self.done, self.sub_reward, grounding = self.webshop_env.step(session_idx, action)
        self.goal = self.webshop_env.goal
    
    def run(self):
        while not self.webshop_env.done:
            self.step()

    def step(self):
        if "[Search]" in self.observation:
            if self.search_query_list:
                instruction = f"""{self.goal} You have searched {self.search_query_list}. Please change to a different query."""
            else:
                instruction = self.goal
            task = TaskPackage(instruction=instruction)
            self.search_agent(task)
            self.observation = self.webshop_env.obs
            print("last obs:", self.observation)
        else:
            instruction = f"""{self.goal}\n{self.observation}"""
            task = TaskPackage(instruction=instruction)
            self.click_agent(task)
            self.observation = self.webshop_env.obs
        