from webshop_env import Webshop
from agentlite.actions import BaseAction

class ClickAction(BaseAction):
    def __init__(self, session_idx, env:Webshop = None) -> None:
        self.session_idx = session_idx
        self.env = env
        action_name = "click"
        action_desc = "click a button in the web page"
        params_doc = {"button": "the name of the button to click"}
        super().__init__(action_name, action_desc, params_doc)

    def __call__(self, button):
        env_action = f"click[{button}]"
        observation, reward, done, sub_reward, grounding = self.env.step(
            self.session_idx, env_action
        )
        if not done:
            return observation
        else:
            return "shopping is finished."


class SearchAction(BaseAction):
    def __init__(self, session_idx, env:Webshop = None) -> None:
        self.session_idx = session_idx
        self.env = env
        action_name = "search"
        action_desc = "search for a product in the webshop"
        params_doc = {"product": "the name of the product to search for"}
        super().__init__(action_name, action_desc, params_doc)

    def __call__(self, product):
        env_action = f"search[{product}]"
        observation, reward, done, sub_reward, grounding = self.env.step(
            self.session_idx, env_action
        )
        return observation
