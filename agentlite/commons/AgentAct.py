from pydantic import BaseModel


class AgentAct(BaseModel):
    """Using AgentAct class to design the agent self-actions and API-call actions

    :param name: action name
    :type name: str
    :param desc: the description/documents of this action
    :type desc: str, optional
    """

    name: str
    desc: str = None
    params: dict = None


ActObsChainType = list[tuple[AgentAct, str]]
