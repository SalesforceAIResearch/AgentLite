# defining the educator class for chat
from typing import List
from agentlite.agents import ABCAgent, BaseAgent, ManagerAgent
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend, OPENAI_CHAT_MODELS, DEEPSEEK_CHAT_MODELS
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.actions.BaseAction import BaseAction

from agentlite.logging.streamlit_logger import UILogger
from agentlite.commons import TaskPackage

import streamlit as st
import os


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

# Header of the app
st.set_page_config(page_title="Philosophers Team", page_icon="🧠", layout="wide")

# 模型选择
st.sidebar.header("模型设置")

api_provider = st.sidebar.selectbox(
    "选择API提供商",
    ["OpenAI", "DeepSeek"],
)

if api_provider == "OpenAI":
    # 确保已设置API密钥
    if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "EMPTY":
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            st.warning("请先在侧边栏输入API密钥")
            st.stop()
            
    llm_model = st.sidebar.selectbox(
        "选择模型",
        OPENAI_CHAT_MODELS,
        index=OPENAI_CHAT_MODELS.index("gpt-4") if "gpt-4" in OPENAI_CHAT_MODELS else 0
    )
    
elif api_provider == "DeepSeek":
    # 确保已设置API密钥
    if "DEEPSEEK_API_KEY" not in os.environ or os.environ["DEEPSEEK_API_KEY"] == "EMPTY":
        deepseek_api_key = st.sidebar.text_input("DeepSeek API Key", type="password")
        if deepseek_api_key:
            os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
        else:
            st.warning("请先在侧边栏输入API密钥")
            st.stop()
            
    llm_model = st.sidebar.selectbox(
        "选择模型",
        DEEPSEEK_CHAT_MODELS,
        index=0
    )

# 根据选择设置llm配置
llm_config = LLMConfig({"llm_name": llm_model, "temperature": 0.0})
llm = get_llm_backend(llm_config)

col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"AgentLite: Philosophers Team (使用{api_provider}的{llm_model})")

    agent_info = [
        {"name": "Socrates", "role": "Expert of Socratic method, ancient Greek philosopher."},
        {"name": "Nietzsche", "role": "Expert of nihilism, German philosopher, cultural critic."},
        {"name": "Aristotle", "role": "Expert of logic and moderation, ancient Greek philosopher."},
        {"name": "Buddha", "role": "Expert of Buddhism, founding figure of the Buddhist tradition."},
        {"name": "Confucius", "role": "Expert of Confucianism, founding figure of the Confucianism."},
    ]

with col2:
    st.info(
        "This is a multi-agent system demonstration. Unlike a simple agent, by bringing multiple philosophers with different backgrounds and approaches into the conversation, we can thoroughly discuss your task from different aspects."
    )

agents = []
logger = UILogger()

for info in agent_info:
    agent = BaseAgent(
        name=info["name"],
        role=info["role"],
        llm=llm,
        logger=logger,
    )
    agents.append(agent)

manager = ManagerAgent(
    llm=llm,
    name="Aristotle",
    role="You are the team's philosopher manager, prioritizing unity in conversation. Your key responsibilities include:\n1. Directing discussions by inviting appropriate team members to share insights based on their expertise.\n2. Delivering final responses that synthesize the team's collective wisdom.\n3. Fostering respectful dialogue where each member's unique perspective is valued.\n4. Maintaining focused conversations by keeping discussions relevant to the task at hand.\n5. Discreetly handle sensitive topics to maintain a professional atmosphere.\nAs manager, you embody Aristotle's principles of moderation and practical wisdom in your leadership approach.",
    TeamAgents=agents,
    logger=logger,
)

# Notes and warnings
st.warning(
    "Note: this service is hosted within Salesforce and user inputs are NOT logged. Regardless, it is not recommended that you enter confidential information."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# Make sure "philosopher_round" exists
if "philosopher_round" not in st.session_state:
    st.session_state.philosopher_round = 0

if prompt := st.chat_input("Enter your task:"):
    task_package = TaskPackage(instruction=prompt)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    res = manager(task_package)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(res)
    # Add message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": res})

st.info(
    "AgentLite is an AI created by the Salesforce AI research group. Your security and confidentiality are prioritized. If you have questions or need assistance with anything, just let me know, and I'll do my best to help. Have a great day!"
)
