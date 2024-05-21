import os

import wikipedia
import duckduckgo_search

from agentlite.actions.BaseAction import BaseAction


class DuckSearch(BaseAction):
    def __init__(self) -> None:
        action_name = "DuckDuckGo_Search"
        action_desc = "Using this action to search online content."
        params_doc = {"query": "the search string. be simple."}
        self.ddgs = duckduckgo_search.DDGS()
        super().__init__(
            action_name=action_name, action_desc=action_desc, params_doc=params_doc,
        )

    def __call__(self, query):
        results = self.ddgs.chat(query)
        return results
    
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
