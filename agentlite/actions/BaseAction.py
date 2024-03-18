from inspect import signature


class BaseAction:
    """
    this is the Action class for agent to use.
    Using this Action class to wrap APIs, tools as an Action of an agent
    BaseAction has three necessary property
    - action_name
    - action_desc
    - params_doc
    Agent will use these three property to understand how to use this action.
    """

    def __init__(
        self,
        action_name: str,
        action_desc: str,
        params_doc: dict = {},
    ) -> None:
        """
        the agent action should be connected with data and env
        Input:
            action_name (str): action_name should be simple and distinctive.
                             One word string, concat with '_' or camel style.
            action_desc (str): agent use action_desc to understand this action
            params_doc (dict): a document to explain the input parameters to the API
        """
        self.action_name = action_name
        self.action_desc = action_desc
        self.params_doc = params_doc

    def __call__(self, **kwargs) -> str:
        """
        implement the Action as
        """
        raise NotImplementedError

    def __get_kwargs__(self):
        return signature(self.__call__)
