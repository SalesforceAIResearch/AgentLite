
from webshop_env import Webshop
from agentlite.actions import BaseAction

webshop_env = Webshop()

class ClickAction(BaseAction):
    def __init__(self, session_idx) -> None:
        self.session_idx = session_idx
        action_name = "click"
        action_desc = "click a button in the web page"
        params_doc = {"button": "the name of the button to click"}
        super().__init__(action_name, action_desc, params_doc)
    
    def __call__(self, button):
        env_action = f"click[{button}]"
        observation, reward, done, sub_reward, grounding  = webshop_env.step(self.session_idx, env_action)
        return observation
    
class SearchAction(BaseAction):
    def __init__(self, session_idx) -> None:
        self.session_idx = session_idx
        action_name = "search"
        action_desc = "search for a product in the webshop"
        params_doc = {"product": "the name of the product to search for"}
        super().__init__(action_name, action_desc, params_doc)
    
    def __call__(self, product):
        env_action = f"search[{product}]"
        observation, reward, done, sub_reward, grounding  = webshop_env.step(self.session_idx, env_action)
        return observation
