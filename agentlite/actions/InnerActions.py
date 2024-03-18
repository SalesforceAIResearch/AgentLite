from agentlite.actions.BaseAction import BaseAction

DEF_INNER_ACT_OBS = "OK"
INNER_ACT_KEY = "response"
REASONING_TYPES = ["react", "act", "planact", "planreact"]


class ThinkAction(BaseAction):
    def __init__(self) -> None:
        action_name = "Think"
        action_desc = "Conduct thinking and reasoning process for solving task."
        params_doc = {
            INNER_ACT_KEY: "this is your thinking response. Be specific and critical."
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, **kwargs):
        return DEF_INNER_ACT_OBS


class FinishAction(BaseAction):
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = "Complete the task with a response."
        params_doc = {
            INNER_ACT_KEY: "this is the finish action response. Respond towards the task instruction."
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, response):
        return response


class PlanAction(BaseAction):
    def __init__(self) -> None:
        action_name = "Plan"
        action_desc = "Plan step-by-step solutions for a task."
        params_doc = {
            INNER_ACT_KEY: "this is the generated plans. Should decompose the task instructions as easy to execute steps."
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, **kwargs):
        return DEF_INNER_ACT_OBS
