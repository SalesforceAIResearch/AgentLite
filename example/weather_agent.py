from weather_actions import get_user_current_date, get_user_current_location, get_latitude_longitude, get_weather_forcast
from agentlite.actions import FinishAct, ThinkAct, PlanAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend, LLMConfig
from agentlite.logging.multi_agent_log import AgentLogger


class WeatherAgent(BaseAgent):
    def __init__(self, llm: BaseLLM, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "WeatherAgent"
        role = "Using weather API to get weather information. Do not skip any steps."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[get_user_current_date(), get_user_current_location(), get_latitude_longitude(), get_weather_forcast()],
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG),
            reasoning_type=agent_arch,
        )

if __name__ == "__main__":
    llm_name = "gpt-4"
    llm_config = LLMConfig({"llm_name": llm_name, "temperature": 0.0})
    llm = get_llm_backend(llm_config)
    agent = WeatherAgent(llm)
    FLAG_CONTINUE = True
    while FLAG_CONTINUE:
        input_text = input("Ask Weather Agent question:\n")
        task = TaskPackage(instruction=input_text)
        agent(task)
        if input("Do you want to continue? (y/n): ") == "n":
            FLAG_CONTINUE = False