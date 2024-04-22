# defining the educator class for chat
from typing import List
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.actions.BaseAction import BaseAction

from agentlite.logging.streamlit_logger import UILogger

import streamlit as st


# Define the Philosopher class
class Philosopher(BaseAgent):
    def __init__(
        self,
        philosopher,
        llm: BaseLLM,
    ):
        name = philosopher
        role = f"""You are {philosopher}, the famous educator in history. You are very familiar with {philosopher}'s Book and Thought. Tell your opinion on behalf of {philosopher}."""
        super().__init__(
            name=name,
            role=role,
            llm=llm,
        )


# import os
# LAM_URL = os.environ["LAM_URL"]
# llm_config = LLMConfig(
#         {
#             "llm_name": "xlam_v2",
#             "temperature": 0.0,
#             "base_url": LAM_URL,
#             "api_key": "EMPTY"
#         }
#     )
llm_name = "gpt-4"
llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
llm = get_llm_backend(llm_config)

logger = UILogger()


Confucius = Philosopher(philosopher="Confucius", llm=llm)
Socrates = Philosopher(philosopher="Socrates", llm=llm)
Aristotle = Philosopher(philosopher="Aristotle", llm=llm)

from agentlite.commons import AgentAct, TaskPackage
from agentlite.actions import ThinkAct, FinishAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents.agent_utils import AGENT_CALL_ARG_KEY

# Add an illustration example for the philosopher agent
exp_task = "What do you think the meaning of life?"
exp_task_pack = TaskPackage(instruction=exp_task)

act_1 = AgentAct(
    name=ThinkAct.action_name,
    params={
        INNER_ACT_KEY: f"""Based on my thought, we are born to live a meaningful life, and it is in living a meaningful life that our existence gains value. Even if a life is brief, if it holds value, it is meaningful. A life without value is merely existence, a mere survival, a walking corpse."""
    },
)
obs_1 = "OK. I have finished my thought, I can pass it to the manager now."

act_2 = AgentAct(
    name=FinishAct.action_name,
    params={INNER_ACT_KEY: "I can summarize my thought now."},
)
obs_2 = "I finished my task, I think the meaning of life is to pursue value for the whold world."
exp_act_obs = [(act_1, obs_1), (act_2, obs_2)]

Confucius.prompt_gen.add_example(task=exp_task_pack, action_chain=exp_act_obs)
Socrates.prompt_gen.add_example(task=exp_task_pack, action_chain=exp_act_obs)
Aristotle.prompt_gen.add_example(task=exp_task_pack, action_chain=exp_act_obs)

# define the manager agent
from agentlite.agents import ManagerAgent

manager_agent_info = {
    "name": "ClockTower",
    "role": "you are managing Confucius, Socrates and Aristotle to discuss on questions. Ask their opinion one by one and summarize their view of point.",
}
team = [Confucius, Socrates, Aristotle]
manager_agent = ManagerAgent(
    name=manager_agent_info["name"],
    role=manager_agent_info["role"],
    llm=llm,
    TeamAgents=team,
    logger=logger,
)

# add illustration example for the manager agent

exp_task = "What is the meaning of life?"
exp_task_pack = TaskPackage(instruction=exp_task)

act_1 = AgentAct(
    name=ThinkAct.action_name,
    params={
        INNER_ACT_KEY: f"""I can ask Confucius, Socrates and Aristotle one by one on their thoughts, and then summary the opinion myself."""
    },
)
obs_1 = "OK."


act_2 = AgentAct(
    name=Confucius.name,
    params={
        AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
    },
)
obs_2 = """Based on my thought, I think the meaning of life is to pursue value for the whold world."""


act_3 = AgentAct(
    name=ThinkAct.action_name,
    params={
        INNER_ACT_KEY: f"""I have obtained information from Confucius, I need to collect more information from Socrates."""
    },
)
obs_3 = "OK."

act_4 = AgentAct(
    name=Socrates.name,
    params={
        AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
    },
)
obs_4 = """I think the meaning of life is finding happiness."""

act_5 = AgentAct(
    name=ThinkAct.action_name,
    params={
        INNER_ACT_KEY: f"""I have obtained information from Confucius and Socrates, I can collect more information from Aristotle."""
    },
)
obs_5 = "OK."

act_6 = AgentAct(
    name=Aristotle.name,
    params={
        AGENT_CALL_ARG_KEY: "What is your opinion on the meaning of life?",
    },
)
obs_6 = """I believe the freedom of spirit is the meaning."""


act_7 = AgentAct(
    name=FinishAct.action_name,
    params={
        INNER_ACT_KEY: "Their thought on the meaning of life is to pursue value, happiniss and freedom of spirit."
    },
)
obs_7 = "Task Completed. The meaning of life is to pursue value, happiness and freedom of spirit."

exp_act_obs = [
    (act_1, obs_1),
    (act_2, obs_2),
    (act_3, obs_3),
    (act_4, obs_4),
    (act_5, obs_5),
    (act_6, obs_6),
    (act_7, obs_7),
]

manager_agent.prompt_gen.add_example(
    task=exp_task_pack, action_chain=exp_act_obs
)

st.set_page_config(page_title="Triad of Wisdom", page_icon="🧠", layout="wide")

# Header of the app
st.header("AgentLite: Triad of Wisdom")
st.sidebar.header("Triad of Wisdom")
st.sidebar.markdown(
    "This is a top-class philosopher group chat that integrates the wisdoms of of Confucius, Socrates and Aristotle to answer your life questions."
)

# Notes and warnings
st.warning(
    "Note: this service is hosted within Salesforce and user inputs are NOT logged. Regardless, it is not recommended that you enter confidential information."
)


# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("Ask Confucius, Socrates and Aristotle a question:"):
    task_pack = TaskPackage(instruction=prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    response = manager_agent(task_pack)
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.info(
    "AgentLite is an AI created by the Salesforce AI research group. Your security and confidentiality are prioritized. If you have questions or need assistance with anything, just let me know, and I'll do my best to help. Have a great day!"
)
