"""functions or objects shared by agents"""

import re

from agentlite.actions.BaseAction import BaseAction


def name_checking(name: str):
    """ensure no white space in name"""
    white_space = [" ", "\n", "\t"]
    for w in white_space:
        if w in name:
            return False
    return True


def act_match(input_act_name: str, act: BaseAction):
    if input_act_name == act.action_name:  # exact match
        return True
    ## To-Do More fuzzy match
    return False


def parse_action(string: str) -> tuple[str, str]:
    """
    Parse an action string into an action type and an argument.
    """

    string = string.strip(" ").strip(".").strip(":").split("\n")[0]
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1).strip()
        argument = match.group(2).strip()
        return action_type, eval(argument)
    else:
        action_type, argument = fuzzy_parse_action(string)
        try:
            argument = eval(argument)
        except:
            print("unable to parse string:", string)
            raise ValueError
        return action_type, eval(argument)


def fuzzy_parse_action(text):
    text = text.strip(" ").strip(".")
    pattern = r"^(\w+)\[(.+)\]"
    match = re.match(pattern, text)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return text, ""


AGENT_CALL_ARG_KEY = "Task"
NO_TEAM_MEMEBER_MESS = (
    """No team member for manager agent. Please check your manager agent team."""
)
ACION_NOT_FOUND_MESS = (
    """"This is the wrong action to call. Please check your available action list."""
)
