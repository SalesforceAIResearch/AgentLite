# get llm backend
from agentlite.llm.agent_llms import get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig

llm_config_dict = {
    "llm_name": "gpt-3.5-turbo-16k-0613",
    "temperature": 0.9,
}
llm_config = LLMConfig(llm_config_dict)
llm = get_llm_backend(llm_config)

# define the individual agents
from SearchAgent import DuckSearchAgent, WikiSearchAgent

wiki_search_agent = WikiSearchAgent(llm)
duck_search_agent = DuckSearchAgent(llm)

# define the manager agent
from agentlite.agents import ManagerAgent

manager_agent_info = {
    "name": "search_manager",
    "role": "you are controlling wiki_search_agent and duck_search_agent to complete the search task. You should first use wiki_search_agent to complete the search task. If didn't answer the task, please try to ask duck_search_agent. You should integrate the answer from both agent to finalize the task.",
}
search_manager = ManagerAgent(
    llm,
    manager_agent_info["name"],
    manager_agent_info["role"],
    TeamAgents=[wiki_search_agent, duck_search_agent],
)

# test the manager agent with TaskPackage
from agentlite.commons import TaskPackage

test_task = "what is salesforce famous for?"
test_task_pack = TaskPackage(instruction=test_task, task_creator="User")
response = search_manager(test_task_pack)
print(response)

# terminal runnning
# Agent search_manager receives the following TaskPackage:
# [
#         Task ID: 6f6bffdd-1ba8-4f7c-b326-8f409865fef0
#         Instruction: what is salesforce famous for?
# ]
# ====search_manager starts execution on TaskPackage 6f6bffdd-1ba8-4f7c-b326-8f409865fef0====
# Agent search_manager takes 0-step Action:
# {
#         name: wiki_search_agent
#         params: {'Task': 'What is salesforce famous for?'}
# }
# Agent wiki_search_agent receives the following TaskPackage:
# [
#         Task ID: 6f6bffdd-1ba8-4f7c-b326-8f409865fef0
#         Instruction: What is salesforce famous for?
# ]
# ====wiki_search_agent starts execution on TaskPackage 6f6bffdd-1ba8-4f7c-b326-8f409865fef0====
# Agent wiki_search_agent takes 0-step Action:
# {
#         name: Wikipedia_Search
#         params: {'query': 'salesforce'}
# }
# Observation: Page: Salesforce
# Summary: Salesforce, Inc. is an American cloud-based software company headquartered[TLDR]
# Agent wiki_search_agent takes 1-step Action:
# {
#         name: Wikipedia_Search
#         params: {'query': 'salesforce'}
# }
# Observation: Page: Salesforce
# Summary: Salesforce, Inc. is an American cloud-based software company headquartered[TLDR]
# Agent wiki_search_agent takes 2-step Action:
# {
#         name: Wikipedia_Search
#         params: {'query': 'salesforce'}
# }
# Observation: Page: Salesforce
# Summary: Salesforce, Inc. is an American cloud-based software company headquartered[TLDR]
# Agent wiki_search_agent takes 3-step Action:
# {
#         name: Finish
#         params: {'response': "Salesforce is famous for being the world's largest enterprise software firm and for providing customer relationship management (CRM) software and applications."}
# }
# Observation: "This is the wrong action to call. Please check your available action list.
# =========wiki_search_agent finish execution. TaskPackage[ID:6f6bffdd-1ba8-4f7c-b326-8f409865fef0] status:
# [
#         completion: completed
#         answer: Salesforce is famous for being the world's largest enterprise software firm and for providing customer relationship management (CRM) software and applications.
# ]
# ==========
# Observation: Salesforce is famous for being the world's largest enterprise software firm and for providing custom[TLDR]
# Agent search_manager takes 1-step Action:
# {
#         name: Finish
#         params: {'response': 'According to Wikipedia, Salesforce is famous for providing customer relationship management (CRM) software and applications.'}
# }
# Observation: Task Completed.
# =========search_manager finish execution. TaskPackage[ID:6f6bffdd-1ba8-4f7c-b326-8f409865fef0] status:
# [
#         completion: completed
#         answer: According to Wikipedia, Salesforce is famous for providing customer relationship management (CRM) software and applications.
# ]
# ==========
# According to Wikipedia, Salesforce is famous for providing customer relationship management (CRM) software and applications.
