import os
import wikipedia

from agentlite.actions.BaseAction import BaseAction

class WikipediaSearch(BaseAction):
    def __init__(self) -> None:
        action_name = "Wikipedia_Search"
        action_desc = "Using this API to search Wiki content."
        params_doc = {"query": "the search string. be simple."}

        super().__init__(
            action_name=action_name, action_desc=action_desc, params_doc=params_doc,
        )

    def __call__(self, query):
        search_results = wikipedia.search(query)
        if not search_results:
            return "No results found."
        article = wikipedia.page(search_results[0])
        return article.summary
