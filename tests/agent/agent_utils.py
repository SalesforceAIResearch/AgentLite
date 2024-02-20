import unittest

from agentlite.actions import ThinkAct
from agentlite.agents.agent_utils import act_match


class Test_act_match(unittest.TestCase):
    def test_1(self):
        act_name = "Think"
        assert act_match(act_name, ThinkAct)
