import json

from agentlite.actions.BaseAction import BaseAction
from agentlite.agents.agent_utils import AGENT_CALL_ARG_KEY
from agentlite.commons import AgentAct, TaskPackage

PROMPT_TASK_KEY = "task"
PROMPT_ACT_OBS_KEY = "act_obs"

REASONING_TYPES = []
PROMPT_TOKENS = {
    "instruction": {"begin": "[Instruction]", "end": "[End of Instruction]"},
    "role": {"begin": "[Role]", "end": "[End of Role]"},
    "constraint": {"begin": "[Constraint]", "end": "[End of Constraint]"},
    "action": {"begin": "[Action_Doc]", "end": "[End of Action_Doc]"},
    "example": {"begin": "[Example]", "end": "[End of Example]"},
    "action_format": {
        "begin": "[ActionFormatExample]",
        "end": "[End of ActionFormatExample]",
    },
    "execution": {"begin": "[Execution]", "end": "[End of Execution]"},
    "team": {"begin": "[Team_Doc]", "end": "[End of Team_Doc]"},
}


CONSTRAITS = {
    "simple": "You generation should be simple and clear.",
}

DEFAULT_PROMPT = {
    "agent_instruction": f"""You are an intelligent agent. You should follow your {PROMPT_TOKENS["role"]['begin']}, {PROMPT_TOKENS["action"]['begin']} to take actions. Your generation should follow the example format. Finish the task as best as you can.""",
    "manager_instruction": f"""You are a manager agent. You can assign a task to those agents in your team. Follow your {PROMPT_TOKENS["role"]['begin']}, {PROMPT_TOKENS["action"]['begin']}, {PROMPT_TOKENS["team"]['begin']} to take actions.""",
    "constraint": f"""{CONSTRAITS["simple"]}""",
    "action_format": "Using the following action format example to generate well formatted actions.\n",
}


def format_act_params_example(actions: list[BaseAction]):
    """
    format the api call parameters with the provided api doc
    """
    act_params_example_str = ""
    for act in actions:
        if not act.params_doc:
            raise KeyError("No API call params doc provided")
        agent_act = AgentAct(name=act.action_name, params=act.params_doc)
        act_str = action_format(agent_act)
        act_params_example_str += act_str
        act_params_example_str += "\n"
    return act_params_example_str


def format_agent_call_example(agents_doc: dict[str, str]):
    """
    format the agent call parameters with the provided agent doc
    """
    agent_call_example_str = ""
    for agent_name in agents_doc:
        params = {AGENT_CALL_ARG_KEY: "Please follow team doc to generate the task"}
        agent_call_act = AgentAct(name=agent_name, params=params)
        agent_call_str = action_format(agent_call_act)
        agent_call_example_str += agent_call_str
        agent_call_example_str += "\n"
    return agent_call_example_str


def action_format(act: AgentAct, action_trigger: bool=True) -> str:
    """unified format the action as a string"""
    str_params = json.dumps(act.params)
    if action_trigger:
        act_str = f"""Action:{act.name}[{str_params}]"""
    # w/o Action trigger
    else:
        act_str = f"""{act.name}[{str_params}]"""
    return act_str


def action_chain_format(action_chain: list[tuple[AgentAct, str]]):
    """Unified format of action generation of inner actions and outer actions"""
    history = ""
    for act, obs in action_chain:
        history += f"""{action_format(act)}\nObservation: {obs}\n"""
    return history


def task_chain_format(task: TaskPackage, action_chain: list[tuple[AgentAct, str]]):
    context = f"Task:{task.instruction}\n"
    context += action_chain_format(action_chain)
    return context