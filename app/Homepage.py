import streamlit as st
import os

from agentlite.llm.agent_llms import get_llm_backend, LLMConfig, OPENAI_CHAT_MODELS, DEEPSEEK_CHAT_MODELS
from agentlite.agents import BaseAgent, ManagerAgent
from agentlite.commons import TaskPackage
from agentlite.logging.streamlit_logger import UILogger
from example.weather_actions import (
    get_user_current_date,
    get_user_current_location,
    get_latitude_longitude,
    get_weather_forcast,
)

######### Page Configuration #########
st.set_page_config(page_title="AgentLite Demos", page_icon="��", layout="wide")

######### Sidebar #########
with st.sidebar.expander("🔑 API Settings", expanded=True):
    api_provider = st.selectbox("API Provider", ["OpenAI", "DeepSeek"])
    if api_provider == "OpenAI":
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            os.environ['OPENAI_API_KEY'] = key
        models = OPENAI_CHAT_MODELS
    else:
        key = st.text_input("DeepSeek API Key", type="password")
        if key:
            os.environ['DEEPSEEK_API_KEY'] = key
        models = DEEPSEEK_CHAT_MODELS

with st.sidebar.expander("💻 Model & Options", expanded=True):
    llm_model = st.selectbox("LLM Model", models)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.9, 0.05)
    demo = st.radio("Select Demo", ["Weather Agent", "Philosophers Team"])

with st.sidebar.expander("⚙️ Chat Controls", expanded=False):
    # Clear per-demo history
    if st.button("Clear Weather History"):
        st.session_state['weather_history'] = []
    if st.button("Clear Philosopher History"):
        st.session_state['philo_history'] = []
######### End Sidebar #########

######### Initialize LLM & Logger #########
llm = get_llm_backend(LLMConfig({'llm_name': llm_model, 'temperature': temperature}))
logger = UILogger()

# Prepare chat histories
if 'weather_history' not in st.session_state:
    st.session_state['weather_history'] = []
if 'philo_history' not in st.session_state:
    st.session_state['philo_history'] = []
######### End Init #########

st.title(f"AgentLite Demo: {demo}")

# Main area: use Tabs to switch demos
tabs = st.tabs(["🌤️ Weather Agent", "🧠 Philosophers Team"])

with tabs[0]:  # Weather Agent
    st.subheader("🌤️ Weather Agent")
    st.info("Get weather information for a location.")
    # Display history
    for msg in st.session_state['weather_history']:
        st.chat_message(msg['role']).markdown(msg['content'])
    # User input
    if prompt := st.chat_input("Your weather request:", key='weather_input'):
        st.session_state['weather_history'].append({'role':'user','content':prompt})
        st.chat_message('user').markdown(prompt)
        # Run agent
        agent = BaseAgent(
            name='Weatherman', role='Fetch weather info', llm=llm,
            actions=[get_user_current_date(), get_user_current_location(), get_latitude_longitude(), get_weather_forcast()],
            reasoning_type='react', logger=logger
        )
        response = agent(TaskPackage(instruction=prompt))
        st.session_state['weather_history'].append({'role':'assistant','content':response})
        st.chat_message('assistant').markdown(response)

with tabs[1]:  # Philosophers Team
    st.subheader("🧠 Philosophers Team")
    st.info("Discuss tasks with a team of philosophers.")
    # Display history
    for msg in st.session_state['philo_history']:
        st.chat_message(msg['role']).markdown(msg['content'])
    # User input
    if prompt := st.chat_input("Your philosophical question:", key='philo_input'):
        st.session_state['philo_history'].append({'role':'user','content':prompt})
        st.chat_message('user').markdown(prompt)
        # Prepare agents
        philosophers = [
            {'name':'Socrates','role':'Expert of Socratic method.'},
            {'name':'Nietzsche','role':'Expert of nihilism.'},
            {'name':'Aristotle','role':'Expert of logic.'},
            {'name':'Buddha','role':'Expert of Buddhism.'},
            {'name':'Confucius','role':'Expert of Confucianism.'}
        ]
        agents = [BaseAgent(name=a['name'],role=a['role'],llm=llm,logger=logger) for a in philosophers]
        manager = ManagerAgent(
            llm=llm, name='PhiloManager', role='Synthesize philosophical insights.',
            TeamAgents=agents, logger=logger
        )
        response = manager(TaskPackage(instruction=prompt))
        st.session_state['philo_history'].append({'role':'assistant','content':response})
        st.chat_message('assistant').markdown(response)
