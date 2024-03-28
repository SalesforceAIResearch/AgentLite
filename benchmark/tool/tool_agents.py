from tool_actions import get_weather_actions, get_academia_actions, get_movie_actions
from agentlite.actions import FinishAct, ThinkAct, PlanAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging.multi_agent_log import AgentLogger

class WeatherAgent(BaseAgent):
    def __init__(self, env, llm: BaseLLM, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "WeatherAgent"
        role = "Using weather API to get weather information. Do not skip any steps."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[],
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.agent_arch = agent_arch
        self.env = env
        self.actions = get_weather_actions(env)
        if agent_arch in ["zs"]:
            pass
        elif agent_arch in ["zst"]:
            self.actions.append(ThinkAct)
        elif agent_arch in ["react"]:
            self.actions.append(ThinkAct)
        elif agent_arch in ["planact"]:
            self.actions.append(PlanAct)
        elif agent_arch in ["planreact"]:
            self.actions.append(PlanAct)
            self.actions.append(ThinkAct)
        self.__build_examples__()
    
    def __build_examples__(self):
        if self.agent_arch in ["react"]:
            self.__build_react_example__()
        if self.agent_arch in ["act"]:
            self.__build_act_example__()
        if self.agent_arch in ["planact"]:
            self.__build_planact_example__()
        if self.agent_arch in ["planreact"]:
            self.__build_planact_example__()
    
    def __build_planact_example__(self):
        goal = "What is the lowest temperature yesterday?"
        task_package = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the lowest temperature yesterday. First, I need to the get the user current location with get_user_current_location action. Then, I need to get the latitude and longitude information of the location via get_latitude_longitude. After getting the location information, I need to get the current date to know the date of yesterday via get_user_current_date action. Next, I can get the historical temperature data of the location in the date of yesterday with get_historical_temp. If I get the minimum temperature data, I can call finish to end the goal and return the lowest value."})
        obs = "Shanghai"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_user_current_location", params={})
        obs = "Shanghai"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_latitude_longitude", params={"name": "Shanghai"})
        obs = "{'results': [{'name': 'Shanghai', 'latitude': 31.22222, 'longitude': 121.45806, 'country_code': 'CN'}, {'name': 'Shanghai', 'latitude': 34.85009, 'longitude': -87.08501, 'country_code': 'US'}, {'name': 'Cornelia', 'latitude': 38.64363, 'longitude': -93.73938, 'country_code': 'US'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_user_current_date", params={})
        obs = "2015-01-02"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_historical_temp", params={"latitude": 31.22222, "longitude": 121.45806, "start_date": "2015-01-01", "end_date": "2015-01-01"})
        obs = "{'latitude': 31.200005, 'longitude': 121.5, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '°C', 'temperature_2m_min': '°C', 'temperature_2m_mean': '°C'}, 'daily': {'time': ['2015-01-01'], 'temperature_2m_max': [4.3], 'temperature_2m_min': [-3.6], 'temperature_2m_mean': [-0.1]}}"
        act_obs.append((act, obs))
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "-3.6"})
        obs = "-3.6"
        act_obs.append((act, obs))
        self.add_example(task=task_package, action_chain=act_obs)
    
    def __build_act_example__(self):
        goal = "What is the lowest temperature yesterday?"
        task_package = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name="get_user_current_location", params={})
        obs = "Shanghai"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_latitude_longitude", params={"name": "Shanghai"})
        obs = "{'results': [{'name': 'Shanghai', 'latitude': 31.22222, 'longitude': 121.45806, 'country_code': 'CN'}, {'name': 'Shanghai', 'latitude': 34.85009, 'longitude': -87.08501, 'country_code': 'US'}, {'name': 'Cornelia', 'latitude': 38.64363, 'longitude': -93.73938, 'country_code': 'US'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_user_current_date", params={})
        obs = "2015-01-02"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_historical_temp", params={"latitude": 31.22222, "longitude": 121.45806, "start_date": "2015-01-01", "end_date": "2015-01-01"})
        obs = "{'latitude': 31.200005, 'longitude': 121.5, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '°C', 'temperature_2m_min': '°C', 'temperature_2m_mean': '°C'}, 'daily': {'time': ['2015-01-01'], 'temperature_2m_max': [4.3], 'temperature_2m_min': [-3.6], 'temperature_2m_mean': [-0.1]}}"
        act_obs.append((act, obs))
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "-3.6"})
        obs = "-3.6"
        act_obs.append((act, obs))
        self.add_example(task=task_package, action_chain=act_obs)
    
    def __build_planreact_example__(self):
        goal = "What is the lowest temperature yesterday?"
        task_package = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the lowest temperature yesterday. First, I need to the get the user current location with get_user_current_location action. Then, I need to get the latitude and longitude information of the location via get_latitude_longitude. After getting the location information, I need to get the current date to know the date of yesterday via get_user_current_date action. Next, I can get the historical temperature data of the location in the date of yesterday with get_historical_temp. If I get the minimum temperature data, I can call finish to end the goal and return the lowest value."})
        obs = "Shanghai"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY:"This question is about the lowest temperature yesterday. I should get the latitude and longitude information of user\'s current location first."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_user_current_location", params={})
        obs = "Shanghai"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_latitude_longitude", params={"name": "Shanghai"})
        obs = "{'results': [{'name': 'Shanghai', 'latitude': 31.22222, 'longitude': 121.45806, 'country_code': 'CN'}, {'name': 'Shanghai', 'latitude': 34.85009, 'longitude': -87.08501, 'country_code': 'US'}, {'name': 'Cornelia', 'latitude': 38.64363, 'longitude': -93.73938, 'country_code': 'US'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY:"I have got the latitude and longitude information of Shanghai, I should get the current date to get the date of yesterday."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_user_current_date", params={})
        obs = "2015-01-02"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "Current date in 2015-01-02, so yesterday is 2015-01-01. Now, I can get the temperature data of Shanghai in 2015-01-01."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_historical_temp", params={"latitude": 31.22222, "longitude": 121.45806, "start_date": "2015-01-01", "end_date": "2015-01-01"})
        obs = "{'latitude': 31.200005, 'longitude': 121.5, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '°C', 'temperature_2m_min': '°C', 'temperature_2m_mean': '°C'}, 'daily': {'time': ['2015-01-01'], 'temperature_2m_max': [4.3], 'temperature_2m_min': [-3.6], 'temperature_2m_mean': [-0.1]}}"
        act_obs.append((act, obs))
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The observation mentions that \'temperature_2m_min\': [-3.6], so the lowest temperature is -3.6. I will call finish to end the goal."})
        obs = "OK"
        act_obs.append((act, obs))
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "-3.6"})
        obs = "-3.6"
        act_obs.append((act, obs))
        self.add_example(task=task_package, action_chain=act_obs)
    
    def __build_react_example__(self):
        goal = "What is the lowest temperature yesterday?"
        task_package = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY:"This question is about the lowest temperature yesterday. I should get the latitude and longitude information of user\'s current location first."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_user_current_location", params={})
        obs = "Shanghai"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_latitude_longitude", params={"name": "Shanghai"})
        obs = "{'results': [{'name': 'Shanghai', 'latitude': 31.22222, 'longitude': 121.45806, 'country_code': 'CN'}, {'name': 'Shanghai', 'latitude': 34.85009, 'longitude': -87.08501, 'country_code': 'US'}, {'name': 'Cornelia', 'latitude': 38.64363, 'longitude': -93.73938, 'country_code': 'US'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY:"I have got the latitude and longitude information of Shanghai, I should get the current date to get the date of yesterday."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_user_current_date", params={})
        obs = "2015-01-02"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "Current date in 2015-01-02, so yesterday is 2015-01-01. Now, I can get the temperature data of Shanghai in 2015-01-01."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_historical_temp", params={"latitude": 31.22222, "longitude": 121.45806, "start_date": "2015-01-01", "end_date": "2015-01-01"})
        obs = "{'latitude': 31.200005, 'longitude': 121.5, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '°C', 'temperature_2m_min': '°C', 'temperature_2m_mean': '°C'}, 'daily': {'time': ['2015-01-01'], 'temperature_2m_max': [4.3], 'temperature_2m_min': [-3.6], 'temperature_2m_mean': [-0.1]}}"
        act_obs.append((act, obs))
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The observation mentions that \'temperature_2m_min\': [-3.6], so the lowest temperature is -3.6. I will call finish to end the goal."})
        obs = "OK"
        act_obs.append((act, obs))
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "-3.6"})
        obs = "-3.6"
        act_obs.append((act, obs))
        self.add_example(task=task_package, action_chain=act_obs)
        

class AcademiaAgent(BaseAgent):
    def __init__(self, env, llm: BaseLLM, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "AcademiaAgent"
        role = "Using actions to help with academic question answering tasks. Do not skip any steps."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[],
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.agent_arch = agent_arch
        self.env = env
        self.actions = get_academia_actions(env)
        if agent_arch in ["zs"]:
            pass
        elif agent_arch in ["zst"]:
            self.actions.append(ThinkAct)
        elif agent_arch in ["react"]:
            self.actions.append(ThinkAct)
        elif agent_arch in ["planact"]:
            self.actions.append(PlanAct)
        elif agent_arch in ["planreact"]:
            self.actions.append(PlanAct)
            self.actions.append(ThinkAct)
    
    def __build_examples__(self):
        if self.agent_arch in ["react"]:
            self.__build_react_example__()
        if self.agent_arch in ["act"]:
            self.__build_act_example__()
        if self.agent_arch in ["planact"]:
            self.__build_planact_example__()
        if self.agent_arch in ["planreact"]:
            self.__build_planreact_example__()
    
    def __build_planact_example__(self):
        goal = "When was the paper \"Learning the Principle of Least Action with Reinforcement Learning\" published?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the publication date of the paper \"Learning the Principle of Least Action with Reinforcement Learning\". First, I need to load the paper net with loadPaperNet action. Then I need to check the paper information with paperNodeCheck action. If I get the publication date, I can call finish to end the goal and return the date."})
        obs = "Learning the Principle of Least Action with Reinforcement Learning"
        act_obs = [(act, obs)]
        # action-observation
        act = AgentAct(name="loadPaperNet", params={})
        obs = "PaperNet is loaded"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name="paperNodeCheck", params={"node": "Learning the Principle of Least Action with Reinforcement Learning"})
        obs = "{'year': 2021, 'venue': 'AAAI Spring Symposium - MLPS', 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference'}"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2021"})
        obs = "2021"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
    
    def __build_act_example__(self):
        goal = "When was the paper \"Learning the Principle of Least Action with Reinforcement Learning\" published?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name="loadPaperNet", params={})
        obs = "PaperNet is loaded"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name="paperNodeCheck", params={"node": "Learning the Principle of Least Action with Reinforcement Learning"})
        obs = "{'year': 2021, 'venue': 'AAAI Spring Symposium - MLPS', 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference'}"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2021"})
        obs = "2021"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
        
    def __build_react_example__(self):
        goal = "When was the paper \"Learning the Principle of Least Action with Reinforcement Learning\" published?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name="loadPaperNet", params={})
        obs = "PaperNet is loaded"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The question is asking the published date of a paper, I should use paperNodeCheck to check the node from the PaperNet in DBLP graph. The paper node is Learning the Principle of Least Action with Reinforcement Learning."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="paperNodeCheck", params={"node": "Learning the Principle of Least Action with Reinforcement Learning"})
        obs = "{'year': 2021, 'venue': 'AAAI Spring Symposium - MLPS', 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference'}"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The published date of the paper is 2021. I can finish this goal"})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2021"})
        obs = "2021"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
    
    def __build_planreact_example__(self):
        goal = "When was the paper \"Learning the Principle of Least Action with Reinforcement Learning\" published?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the publication date of the paper \"Learning the Principle of Least Action with Reinforcement Learning\". First, I need to load the paper net with loadPaperNet action. Then I need to check the paper information with paperNodeCheck action. If I get the publication date, I can call finish to end the goal and return the date."})
        obs = "Learning the Principle of Least Action with Reinforcement Learning"
        act_obs = [(act, obs)]
        # action-observation
        act = AgentAct(name="loadPaperNet", params={})
        obs = "PaperNet is loaded"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The question is asking the published date of a paper, I should use paperNodeCheck to check the node from the PaperNet in DBLP graph. The paper node is Learning the Principle of Least Action with Reinforcement Learning."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="paperNodeCheck", params={"node": "Learning the Principle of Least Action with Reinforcement Learning"})
        obs = "{'year': 2021, 'venue': 'AAAI Spring Symposium - MLPS', 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference'}"
        act_obs.append((act, obs))  
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The published date of the paper is 2021. I can finish this goal"})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2021"})
        obs = "2021"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
        
class MovieAgent(BaseAgent):
    def __init__(self, env, llm: BaseLLM, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "MovieAgent"
        role = "Using actions to help with movie related tasks. Do not skip any steps."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[],
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.agent_arch = agent_arch
        self.env = env
        self.actions = get_movie_actions(env)
        if agent_arch in ["zs"]:
            pass
        elif agent_arch in ["zst"]:
            self.actions.append(ThinkAct)
        elif agent_arch in ["react"]:
            self.actions.append(ThinkAct)
        elif agent_arch in ["planact"]:
            self.actions.append(PlanAct)
        elif agent_arch in ["planreact"]:
            self.actions.append(PlanAct)
            self.actions.append(ThinkAct)
    
    def __build_examples__(self):
        if self.agent_arch in ["react"]:
            self.__build_react_example__()
        if self.agent_arch in ["act"]:
            self.__build_act_example__()
        if self.agent_arch in ["planact"]:
            self.__build_planact_example__()
        if self.agent_arch in ["planreact"]:
            self.__build_planreact_example__()
    
    def __build_react_example__(self):
        goal = "When did the movie Scream 6 come out?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "This question is about the release date of the movie Scream 6. I need to get the movie id first and use the movie id to query movie details. To get the movie id, I can use get_search_movie action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_search_movie", params={"movie_name": "Scream 6"})
        obs = "{'id': 934433, 'overview': 'Following the latest Ghostface killings, the four survivors leave Woodsboro behind and start a fresh chapter.', 'title': 'Scream VI'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_movie_details", params={"movie_id": "934433"})
        obs = "{'budget': 35000000, 'genres': [{'id': 27, 'name': 'Horror'}, {'id': 53, 'name': 'Thriller'}, {'id': 9648, 'name': 'Mystery'}], 'revenue': 168961389, 'vote_average': 7.175, 'release_date': '2023-03-08'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The release date is 2023-03-08, I will call finish to end this goal"})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2023-03-08"})
        obs = "2023-03-08"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
    
    def __build_planreact_example__(self):
        goal = "When did the movie Scream 6 come out?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the release date of the movie Scream 6. I need to get the movie id first with get_search_movie. Then, I will use the movie id to query movie details with get_movie_details. If I get the release date, I can call finish to end the goal and return the date."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "This question is about the release date of the movie Scream 6. I need to get the movie id first and use the movie id to query movie details. To get the movie id, I can use get_search_movie action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_search_movie", params={"movie_name": "Scream 6"})
        obs = "{'id': 934433, 'overview': 'Following the latest Ghostface killings, the four survivors leave Woodsboro behind and start a fresh chapter.', 'title': 'Scream VI'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_movie_details", params={"movie_id": "934433"})
        obs = "{'budget': 35000000, 'genres': [{'id': 27, 'name': 'Horror'}, {'id': 53, 'name': 'Thriller'}, {'id': 9648, 'name': 'Mystery'}], 'revenue': 168961389, 'vote_average': 7.175, 'release_date': '2023-03-08'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The release date is 2023-03-08, I will call finish to end this goal"})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2023-03-08"})
        obs = "2023-03-08"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
    
    
    def __build_act_example__(self):
        goal = "When did the movie Scream 6 come out?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name="get_search_movie", params={"movie_name": "Scream 6"})
        obs = "{'id': 934433, 'overview': 'Following the latest Ghostface killings, the four survivors leave Woodsboro behind and start a fresh chapter.', 'title': 'Scream VI'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_movie_details", params={"movie_id": "934433"})
        obs = "{'budget': 35000000, 'genres': [{'id': 27, 'name': 'Horror'}, {'id': 53, 'name': 'Thriller'}, {'id': 9648, 'name': 'Mystery'}], 'revenue': 168961389, 'vote_average': 7.175, 'release_date': '2023-03-08'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2023-03-08"})
        obs = "2023-03-08"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
    
    def __build_planact_example__(self):
        goal = "When did the movie Scream 6 come out?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the release date of the movie Scream 6. I need to get the movie id first with get_search_movie. Then, I will use the movie id to query movie details with get_movie_details. If I get the release date, I can call finish to end the goal and return the date."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_search_movie", params={"movie_name": "Scream 6"})
        obs = "{'id': 934433, 'overview': 'Following the latest Ghostface killings, the four survivors leave Woodsboro behind and start a fresh chapter.', 'title': 'Scream VI'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_movie_details", params={"movie_id": "934433"})
        obs = "{'budget': 35000000, 'genres': [{'id': 27, 'name': 'Horror'}, {'id': 53, 'name': 'Thriller'}, {'id': 9648, 'name': 'Mystery'}], 'revenue': 168961389, 'vote_average': 7.175, 'release_date': '2023-03-08'}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "2023-03-08"})
        obs = "2023-03-08"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
