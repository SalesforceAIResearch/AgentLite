import time

from agentlite.commons.TaskPackage import TaskPackage

from .agent_utils import name_checking


class ABCAgent:
    """this is the abstract class for agent. Only the name and role are required.
    The calling methods should be developed by developers.

    :param name: short name for agent, use it for id part
    :type name: str
    :param role: describe the job duty of this agent
    :type role: str
    """

    def __init__(self, name: str, role: str):
        """
        initialization
        """
        assert name_checking(
            name
        ), "only one term should be given for the name. No white space."
        self.name = name  #
        self.role = role  #
        self.join_time = time.time()
        self.id = f"{self.name}_{self.join_time}"

    def __call__(self, task: TaskPackage) -> str:
        """agent can be called with a task. it will assign the task and then execute and respond

        :param task: the task agent to solve
        :type task: TaskPackage
        :raises NotImplementedError: _description_
        :return: the response of this task
        :rtype: str
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """Get the name of the agent."""
        return self.name

    def get_role(self) -> str:
        """Describe the agent role."""
        return self.role
