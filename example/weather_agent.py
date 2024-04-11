import os 

from weather_actions import get_user_current_date, get_user_current_location, get_latitude_longitude, get_weather_forcast

from agentlite.actions import FinishAct, ThinkAct, PlanAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend, LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger

# LAM_URL = os.environ["LAM_URL"]
# print(LAM_URL)
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
agent_info = {
    "name": "WeatherAgent",
    "role": "Using weather API to get weather information. Do not skip any steps.",
}
agent_actions = [get_user_current_date(), get_user_current_location(), get_latitude_longitude(), get_weather_forcast()]
agent = BaseAgent(
    name=agent_info["name"],
    role=agent_info["role"],
    llm=llm,
    actions=agent_actions,
    reasoning_type="react",
)
FLAG_CONTINUE = True
while FLAG_CONTINUE:
    input_text = input("Ask Weather Agent question:\n")
    task = TaskPackage(instruction=input_text)
    agent(task)
    if input("Do you want to continue? (y/n): ") == "n":
        FLAG_CONTINUE = False