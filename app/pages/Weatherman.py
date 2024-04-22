import streamlit as st
import os

from example.weather_actions import (
    get_user_current_date,
    get_user_current_location,
    get_latitude_longitude,
    get_weather_forcast,
)

from agentlite.agents import BaseAgent
from agentlite.commons import TaskPackage
from agentlite.llm.agent_llms import get_llm_backend, LLMConfig

from agentlite.logging.streamlit_logger import UILogger

llm_name = "gpt-4"
llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
llm = get_llm_backend(llm_config)

agent_info = {
    "name": "Weatherman",
    "role": "Using weather API to get weather information. Do not skip any steps.",
}

agent_actions = [
    get_user_current_date(),
    get_user_current_location(),
    get_latitude_longitude(),
    get_weather_forcast(),
]

logger = UILogger()

agent = BaseAgent(
    name=agent_info["name"],
    role=agent_info["role"],
    llm=llm,
    actions=agent_actions,
    reasoning_type="react",
    logger=logger,
)

st.set_page_config(page_title="Weather Agent", page_icon="🌤️", layout="wide")

# Header of the app
st.header("AgentLite: Weather Agent")
st.sidebar.header("Weather Agent")
st.sidebar.markdown(
    "This is a weather agent that uses the Weather API to get weather information on a particular location."
)

# Notes and warnings
st.warning(
    "Note: this service is hosted within Salesforce and user inputs are NOT logged. Regardless, it is not recommended that you enter confidential information."
)


# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

if prompt := st.chat_input("Enter your task:"):
    task_pack = TaskPackage(instruction=prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    response = agent(task_pack)
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.info(
    "AgentLite is an AI created by the Salesforce AI research group. Your security and confidentiality are prioritized. If you have questions or need assistance with anything, just let me know, and I'll do my best to help. Have a great day!"
)
