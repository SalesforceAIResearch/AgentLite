from tool_operation_actions import get_todo_actions, get_sheet_actions
from agentlite.actions import FinishAct, ThinkAct, PlanAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging.terminal_logger import AgentLogger

class TodoAgent(BaseAgent):
    def __init__(self, env, llm: BaseLLM, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "TodoAgent"
        role = "Using actions to help with todo list related tasks. Do not skip any steps."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[],
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.agent_arch = agent_arch
        self.env = env
        self.actions = get_todo_actions(env)
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
        goal = "What is the next task in my todo list?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "This question is about the next task in the todo list. I need to get the todo list first with get_todo_list action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_todo_list", params={})
        obs = "{'tasks': [{'task': 'Buy milk', 'status': 'done'}, {'task': 'Finish the report', 'status': 'done'}, {'task': 'Call mom', 'status': 'todo'}, {'task': 'Go to gym', 'status': 'todo'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The next task is 'Call mom'. I will call finish to end this goal."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Call mom"})
        obs = "Call mom"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)

    def __build_planreact_example__(self):
        goal = "What is the next task in my todo list?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the next task in the todo list. I need to get the todo list first with get_todo_list action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "This question is about the next task in the todo list. I need to get the todo list first with get_todo_list action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_todo_list", params={})
        obs = "{'tasks': [{'task': 'Buy milk', 'status': 'done'}, {'task': 'Finish the report', 'status': 'done'}, {'task': 'Call mom', 'status': 'todo'}, {'task': 'Go to gym', 'status': 'todo'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The next task is 'Call mom'. I will call finish to end this goal."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Call mom"})
        obs = "Call mom"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)

    def __build_act_example__(self):
        goal = "What is the next task in my todo list?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name="get_todo_list", params={})
        obs = "{'tasks': [{'task': 'Buy milk', 'status': 'done'}, {'task': 'Finish the report', 'status': 'done'}, {'task': 'Call mom', 'status': 'todo'}, {'task': 'Go to gym', 'status': 'todo'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Call mom"})
        obs = "Call mom"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)

    def __build_planact_example__(self):
        goal = "What is the next task in my todo list?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the next task in the todo list. I need to get the todo list first with get_todo_list action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_todo_list", params={})
        obs = "{'tasks': [{'task': 'Buy milk', 'status': 'done'}, {'task': 'Finish the report', 'status': 'done'}, {'task': 'Call mom', 'status': 'todo'}, {'task': 'Go to gym', 'status': 'todo'}]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Call mom"})
        obs = "Call mom"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)

class SheetAgent(BaseAgent):
    def __init__(self, env, llm: BaseLLM, agent_arch: str = "react", PROMPT_DEBUG_FLAG=False):
        name = "SheetAgent"
        role = "Using actions to help with spreadsheet related tasks. Do not skip any steps."
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=[],
            logger=AgentLogger(PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG)
        )
        self.agent_arch = agent_arch
        self.env = env
        self.actions = get_sheet_actions(env)
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
        goal = "What is the sum of column A in the sheet?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "This question is about the sum of column A in the sheet. I need to get the sheet data first with get_sheet_data action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_sheet_data", params={})
        obs = "{'columns': ['A', 'B', 'C'], 'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_column_sum", params={"column": "A"})
        obs = "12"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "12"})
        obs = "12"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)

    def __build_planreact_example__(self):
        goal = "What is the sum of column A in the sheet?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the sum of column A in the sheet. I need to get the sheet data first with get_sheet_data action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "This question is about the sum of column A in the sheet. I need to get the sheet data first with get_sheet_data action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_sheet_data", params={})
        obs = "{'columns': ['A', 'B', 'C'], 'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_column_sum", params={"column": "A"})
        obs = "12"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "12"})
        obs = "12"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)
    
    def __build_act_example__(self):
        goal = "What is the sum of column A in the sheet?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name="get_sheet_data", params={})
        obs = "{'columns': ['A', 'B', 'C'], 'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_column_sum", params={"column": "A"})
        obs = "12"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "12"})
        obs = "12"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)

    def __build_planact_example__(self):
        goal = "What is the sum of column A in the sheet?"
        task = TaskPackage(instruction=goal)
        act_obs = []
        # action-observation
        act = AgentAct(name=PlanAct.action_name, params={INNER_ACT_KEY: "This question is about the sum of column A in the sheet. I need to get the sheet data first with get_sheet_data action."})
        obs = "OK"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_sheet_data", params={})
        obs = "{'columns': ['A', 'B', 'C'], 'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name="get_column_sum", params={"column": "A"})
        obs = "12"
        act_obs.append((act, obs))
        # action-observation
        act = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "12"})
        obs = "12"
        act_obs.append((act, obs))
        self.add_example(task=task, action_chain=act_obs)



        