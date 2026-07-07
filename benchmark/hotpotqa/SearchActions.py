import wikipedia

from agentlite.actions.BaseAction import BaseAction


WIKIPEDIA_ACTION_NAME = "Wikipedia_Search"


class WikipediaSearch(BaseAction):
    def __init__(
        self,
        mode: str = "online",
        rag_index_dir: str = None,
        rag_top_k: int = None,
        rag_strategy: str = None,
        rag_embed_model: str = None,
        rag_embedding_dim: int = None,
        rag_cache_dir: str = None,
        rag_device: str = None,
        rag_text_lookup: str = None,
    ) -> None:
        action_name = WIKIPEDIA_ACTION_NAME
        self.mode = mode.lower()
        self.rag_top_k = int(rag_top_k) if rag_top_k is not None else None
        self.retriever = None

        if self.mode == "rag":
            from local_wiki_retriever import LocalWikipediaRetriever

            required_params = {
                "rag_index_dir": rag_index_dir,
                "rag_top_k": rag_top_k,
                "rag_strategy": rag_strategy,
                "rag_embed_model": rag_embed_model,
                "rag_embedding_dim": rag_embedding_dim,
                "rag_device": rag_device,
                "rag_text_lookup": rag_text_lookup,
            }
            missing_params = [
                name for name, value in required_params.items() if value is None
            ]
            if missing_params:
                raise ValueError(
                    "RAG mode requires explicit parameters: "
                    + ", ".join(missing_params)
                )

            self.retriever = LocalWikipediaRetriever(
                index_dir=rag_index_dir,
                top_k=self.rag_top_k,
                strategy=rag_strategy,
                embed_model_name=rag_embed_model,
                embedding_dim=int(rag_embedding_dim),
                device=rag_device,
                text_lookup=rag_text_lookup,
                cache_dir=rag_cache_dir,
            )
            action_desc = "Search local Wikipedia corpus with RAG."
        elif self.mode == "online":
            action_desc = "Using this API to search Wiki content."
        else:
            raise ValueError(f"Unsupported WikipediaSearch mode: {self.mode}")

        params_doc = {"query": "the search string. be simple."}
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, query):
        if self.mode == "rag":
            return self.retriever.format_results(query, top_k=self.rag_top_k)

        search_results = wikipedia.search(query)
        if not search_results:
            return "No results found."
        article = wikipedia.page(search_results[0])
        return article.summary
