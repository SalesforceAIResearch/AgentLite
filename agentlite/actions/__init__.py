from .BaseAction import BaseAction
from .InnerActions import FinishAction, PlanAction, ThinkAction

ThinkAct = ThinkAction()
FinishAct = FinishAction()
PlanAct = PlanAction()
INNER_ACTIONS = [ThinkAct, PlanAct, FinishAct]
