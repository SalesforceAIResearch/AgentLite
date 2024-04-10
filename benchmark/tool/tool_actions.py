from agentlite.actions.BaseAction import BaseAction
from agentlite.actions import FinishAct 

class finish(BaseAction):
    def __init__(self, env=None) -> None:
        action_name = FinishAct.action_name # substitute with the BaseAgent finish action with this new action
        action_desc = "Return an answer and finish the task"
        params_doc = {
            "response": "this is the finish action response. As simple as possible."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)

    def __call__(self, response):
        params = {"answer": response}
        action = (self.action_name, params)
        observation, reward, done, _ = self.env.step(action)
        return observation

# weather actions
class get_user_current_date(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_user_current_date"
        action_desc = "get the current date of the user"
        params_doc = {
            "None": "No input required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_user_current_location(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_user_current_location"
        action_desc = "get the current location of the user"
        params_doc = {
            "None": "No input required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_historical_temp(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_historical_temp"
        action_desc = "get the historical temperature of the user"
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "start_date": "(Type: string) The start date of the historical data (YYYY-MM-DD).",
            "end_date": "(Type: string) The end date of the historical data (YYYY-MM-DD)",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_historical_rain(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_historical_rain"
        action_desc = "Get historical rainfall data for a specified location and date range."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "start_date": "(Type: string) The start date of the historical data (YYYY-MM-DD).",
            "end_date": "(Type: string) The end date of the historical data (YYYY-MM-DD)",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_historical_snow(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_historical_snow"
        action_desc = "Get historical snowfall data for a specified location and date range."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "start_date": "(Type: string) The start date of the historical data (YYYY-MM-DD).",
            "end_date": "(Type: string) The end date of the historical data (YYYY-MM-DD)",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_snow_forecast(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_snow_forecast"
        action_desc = "Get snowfall forecast data for a specified location and date range."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "start_date": "(Type: string) The start date of the historical data (YYYY-MM-DD).",
            "end_date": "(Type: string) The end date of the historical data (YYYY-MM-DD)",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_current_snow(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_current_snow"
        action_desc = "Get current snowfall data for a specified location and date."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "current_date": "(Type: string) The current date to retrieve snowfall data (YYYY-MM-DD).",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_current_temp(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_current_temp"
        action_desc = "Get current temperature data for a specified location and date."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "current_date": "(Type: string) The current date to retrieve snowfall data (YYYY-MM-DD).",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_latitude_longitude(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_latitude_longitude"
        action_desc = "Get latitude and longitude information for a specified location name."
        params_doc = {
            "name": "(Type: string): The name of the location. (e.g., city name)"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_elevation(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_elevation"
        action_desc = "Get elevation data for a specified location."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_temp_forecast(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_temp_forecast"
        action_desc = "Get temperature forecast data for a specified location and date range."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "start_date": "(Type: string) The start date of the historical data (YYYY-MM-DD).",
            "end_date": "(Type: string) The end date of the historical data (YYYY-MM-DD)",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_rain_forecast(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_rain_forecast"
        action_desc = "Get rainfall forecast data for a specified location and date range."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "start_date": "(Type: string) The start date of the historical data (YYYY-MM-DD).",
            "end_date": "(Type: string) The end date of the historical data (YYYY-MM-DD)",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_current_rain(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_current_rain"
        action_desc = "Get current rainfall data for a specified location and date."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "current_date": "(Type: string) The current date to retrieve snowfall data (YYYY-MM-DD).",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_distance(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_distance"
        action_desc = "Calculate the distance between two sets of latitude and longitude coordinates."
        params_doc = {
            "latitude1":" (Type: number): The latitude of the first location.",
            "longitude1": "(Type: number): The longitude of the first location.",
            "latitude2": "(Type: number): The latitude of the second location.",
            "longitude2": "(Type: number): The longitude of the second location."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_historical_air_quality_index(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_distance"
        action_desc = "Get historical air quality index data for a specified location and date range."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "start_date": "(Type: string) The start date of the historical data (YYYY-MM-DD).",
            "end_date": "(Type: string) The end date of the historical data (YYYY-MM-DD)",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_current_air_quality_index(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_current_air_quality_index"
        action_desc = "Get current air quality index data for a specified location and date."
        params_doc = {
            "latitude": "(Type: number) latitude of the location",
            "longitude": "(Type: number) longitude of the location",
            "current_date": "(Type: string) The current date to retrieve snowfall data (YYYY-MM-DD).",
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_air_quality_level(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_air_quality_level"
        action_desc = "Determine the air quality level based on the air quality index (AQI)."
        params_doc = {
            "air_quality_index": "(Type: number): The air quality index (AQI) value."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

# academia actions
class loadPaperNet(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "loadPaperNet"
        action_desc = "Load PaperNet. In this net, nodes are papers and edges are citation relationships between papers."
        params_doc = {
            "None": "No parameter required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class loadAuthorNet(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "loadAuthorNet"
        action_desc = "Load AuthorNet. In this net, nodes are authors and edges are collaboration relationships between authors."
        params_doc = {
            "None": "No parameter required"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class neighbourCheck(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "neighbourCheck"
        action_desc = "List the first-order neighbors connect to the node. In paperNet, neigbours are cited papers of the paper. In authorNet, neigbours are collaborators of the author."
        params_doc = {
            "graph": "(Type: string, Enum: [PaperNet, AuthorNet]): The name of the graph to check",
            "node": "(Type: string): The node for which neighbors will be listed"
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class paperNodeCheck(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "paperNodeCheck"
        action_desc = "Return detailed attribute information of a specified paper in PaperNet. Returns:(authors,year,venue, n_citation,keywords,doc_type)"
        params_doc = {
            "node": "(Type: string): Name of the paper."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class authorNodeCheck(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "authorNodeCheck"
        action_desc = "Return detailed attribute information of a specified author in AuthorNet. Returns:(name, org)"
        params_doc = {
            "node": "(Type: string): Name of the author."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class authorEdgeCheck(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "authorEdgeCheck"
        action_desc = "Return detailed attribute information of the edge between two specified nodes in a AuthorNet."
        params_doc = {
            "node1": "(Type: string): Name of the author.",
            "node2": "(Type: string): Name of the author."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class paperEdgeCheck(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "paperEdgeCheck"
        action_desc = "Return detailed attribute information of the edge between two specified nodes in a PaperNet."
        params_doc = {
            "node1": "(Type: string): Name of the paper.",
            "node2": "(Type: string): Name of the paper."
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_search_movie(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_search_movie"
        action_desc = """Search for a movie by name and return basic details. 
        Returns: (id, overview, title)"""
        params_doc = {
            "movie_name": "(Type: string): The name of the movie to search for.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_details(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_details"
        action_desc = """Get detailed information about a movie by ID. Returns: (budget,genres,revenue,vote_average,release_date)"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_production_companies(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_production_companies"
        action_desc = """Get the production companies of a movie by its ID. Returns: (production_companies)"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_production_countries(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_production_countries"
        action_desc = """Get the production countries of a movie by its ID. Returns: (production_countries)"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_cast(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_production_countries"
        action_desc = """Retrieve the list of the top 10 cast members from a movie by its ID. Returns: (cast : List of the top 10 cast members.)"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_crew(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_crew"
        action_desc = """Retrieve the list of crew members (limited to 10) from a movie by its ID. The list primarily includes Director, Producer, and Writer roles. Returns: (crew : List of the top 10 of crew members)"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_keywords(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_keywords"
        action_desc = """Get the keywords associated with a movie by ID. Returns: keywords"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_search_person(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_search_person"
        action_desc = """Search for a person by name. Returns: (id, name)"""
        params_doc = {
            "person_name": "(Type: string): The name of the person to search for.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_person_details(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_person_details"
        action_desc = """Get detailed information about a person by ID. Returns: (biography, birthday, place_of_birth)"""
        params_doc = {
            "person_id": "(Type: string): The ID of the person.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_person_cast(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_person_cast"
        action_desc = """Retrieve the top 10 movie cast roles of a person by their ID. Returns: (cast : A list of movies where the person has acted, limited to top 10)"""
        params_doc = {
            "person_id": "(Type: string): The ID of the person.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_person_crew(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_person_crew"
        action_desc = """Retrieve the top 10 movie crew roles of a person by their ID. Returns: (crew : A list of movies where the person has participated as crew, limited to top 10)"""
        params_doc = {
            "person_id": "(Type: string): The ID of the person.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation
    
class get_person_external_ids(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_person_external_ids"
        action_desc = """Get the external ids for a person by ID. Returns: (imdb_id, facebook_id, instagram_id, twitter_id)"""
        params_doc = {
            "person_id": "(Type: string): The ID of the person.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_alternative_titles(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_alternative_titles"
        action_desc = """Get the alternative titles for a movie by ID. Returns: (titles, id)"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

class get_movie_translation(BaseAction):
    def __init__(self, env = None) -> None:
        action_name = "get_movie_translation"
        action_desc = """Get the description translation for a movie by ID. Returns: (translations, id)"""
        params_doc = {
            "movie_id": "(Type: string): The ID of the movie.", 
        }
        self.env = env
        super().__init__(action_name, action_desc, params_doc)
        
    def __call__(self, **kwargs):
        action = (self.action_name, kwargs)
        observation, reward, done, _ = self.env.step(action)
        return observation

def get_academia_actions(env):
    academia_actions = [
        loadPaperNet(env),
        loadAuthorNet(env),
        neighbourCheck(env),
        authorNodeCheck(env),
        paperNodeCheck(env),
        authorEdgeCheck(env),
        paperEdgeCheck(env),
        finish(env)
    ]
    return academia_actions

def get_movie_actions(env):
    movie_actions = [
        get_search_movie(env),
        get_movie_details(env),
        get_movie_production_companies(env),
        get_movie_production_countries(env),
        get_movie_keywords(env),
        get_search_person(env),
        get_person_details(env),
        get_movie_cast(env),
        get_movie_crew(env),
        get_person_cast(env),
        get_person_crew(env),
        get_person_external_ids(env),
        get_movie_alternative_titles(env),
        get_movie_translation(env),
        finish(env)
    ]
    return movie_actions

def get_weather_actions(env):
    weather_actions = [
        finish(env),
        get_user_current_date(env),
        get_user_current_location(env),
        get_historical_temp(env),
        get_historical_rain(env),
        get_historical_snow(env),
        get_snow_forecast(env),
        get_snow_forecast(env),
        get_current_snow(env),
        get_current_temp(env),
        get_latitude_longitude(env),
        get_elevation(env),
        get_temp_forecast(env),
        get_rain_forecast(env),
        get_current_rain(env),
        get_distance(env),
        get_historical_air_quality_index(env),
        get_current_air_quality_index(env),
        get_air_quality_level(env)
    ]
    return weather_actions