import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class RetrievalResult:
    node_id: str
    text: str
    score: float
    metadata: Dict[str, str]


#具体的检索方法，ivf/hnsw/enns/encoder
class RetrievalStrategy:
    name = "base"

    def retrieve(self, query_embedding, document_embeddings, top_k: int):
        raise NotImplementedError


class CosineSimilarityStrategy(RetrievalStrategy):
    name = "cosine"

    def retrieve(self, query_embedding, document_embeddings, top_k: int):
        import torch
        import torch.nn.functional as F

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        query_embedding = F.normalize(query_embedding.to(torch.float32), p=2, dim=1)
        document_embeddings = F.normalize(document_embeddings.to(torch.float32), p=2, dim=1)
        scores = torch.matmul(document_embeddings, query_embedding.squeeze(0))
        top_k = min(top_k, scores.shape[0])
        values, indices = torch.topk(scores, k=top_k)
        return indices.cpu().tolist(), values.cpu().tolist()


def build_strategy(name: str) -> RetrievalStrategy:
    if not name:
        raise ValueError("Retrieval strategy must be provided.")
    normalized = name.lower()
    if normalized == "cosine":
        return CosineSimilarityStrategy()
    raise ValueError(f"Unsupported retrieval strategy: {name}")


class LocalWikipediaRetriever:
    """Local Wikipedia retriever backed by precomputed embeddings.

    The default on-disk format matches the TurboRAG builder outputs:
    - codex_vector_cache_float16.pt: {"node_ids": [...], "embs": tensor}
    - codex_node_text_cache.pt: {node_id: text}
    - codex_node_metadata.jsonl: one metadata dict per node

    Retrieval is strategy-based so cosine can later be swapped for FAISS/HNSW
    or two-stage retrieval without touching AgentLite actions.
    """

    def __init__(
        self,
        index_dir: str,
        top_k: int,
        strategy: str,
        embed_model_name: str,
        embedding_dim: int,
        device: str,
        text_lookup: str,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.top_k = int(top_k)
        self.strategy = build_strategy(strategy)
        self.embed_model_name = embed_model_name
        self.embedding_dim = int(embedding_dim)
        self.cache_dir = cache_dir
        self.device = device
        self.text_lookup = text_lookup

        self._torch = None
        self._embedder = None
        self._node_ids: List[str] = []
        self._embeddings = None
        self._metadata_by_id: Optional[Dict[str, Dict[str, str]]] = None
        self._text_cache: Optional[Dict[str, str]] = None

        self._load_vectors()

    def _require_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "RAG mode requires torch. Install it in the agentlite environment "
                "before using WikipediaSearch(mode='rag')."
            ) from exc
        self._torch = torch
        return torch

    def _load_vectors(self) -> None:
        torch = self._require_torch()
        vector_path = self.index_dir / "codex_vector_cache_float16.pt"
        if not vector_path.is_file():
            raise FileNotFoundError(f"Missing vector cache: {vector_path}")
        vector_cache = torch.load(vector_path, map_location=self.device, weights_only=False)
        self._node_ids = list(vector_cache["node_ids"])
        self._embeddings = vector_cache["embs"].to(self.device)

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            import torch
            import torch.nn.functional as F
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "RAG mode requires sentence-transformers and torch to embed queries. "
                "Install them in the agentlite environment before using local RAG."
            ) from exc

        model = SentenceTransformer(
            self.embed_model_name,
            trust_remote_code=True,
            cache_folder=self.cache_dir,
        )
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = 8192
        self._embedder = (model, torch, F)
        return self._embedder

    def _embed_query(self, query: str):
        model, torch, F = self._load_embedder()
        normalized_name = self.embed_model_name.lower()
        if "nomic-embed-text-v1.5" in normalized_name:
            texts = [f"search_query: {query}"]
            embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            embeddings = embeddings.to(torch.float32)
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, : self.embedding_dim]
            return F.normalize(embeddings, p=2, dim=1).to(self.device)
        if "jina-embeddings-v3" in normalized_name:
            embeddings = model.encode(
                [query],
                task="retrieval.query",
                prompt_name="retrieval.query",
                truncate_dim=self.embedding_dim,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            embeddings = embeddings.to(torch.float32)
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, : self.embedding_dim]
            return F.normalize(embeddings, p=2, dim=1).to(self.device)
        raise ValueError(f"Unsupported embedding model: {self.embed_model_name}")

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        if self._metadata_by_id is not None:
            return self._metadata_by_id
        metadata_by_id: Dict[str, Dict[str, str]] = {}
        path = self.index_dir / "codex_node_metadata.jsonl"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    node_id = str(item.get("node_id") or "")
                    if node_id:
                        metadata_by_id[node_id] = item
        self._metadata_by_id = metadata_by_id
        return metadata_by_id

    def _load_text_cache(self) -> Dict[str, str]:
        if self._text_cache is not None:
            return self._text_cache
        torch = self._require_torch()
        path = self.index_dir / "codex_node_text_cache.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing text cache: {path}")
        self._text_cache = torch.load(path, map_location="cpu", weights_only=False)
        return self._text_cache

    def _load_texts_from_shards(self, node_ids: Sequence[str]) -> Dict[str, str]:
        torch = self._require_torch()
        pending = set(node_ids)
        found: Dict[str, str] = {}
        shards_dir = self.index_dir / "shards"
        for shard_path in sorted(shards_dir.glob("shard_*.pt")):
            if not pending:
                break
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for node_id, text in zip(shard.get("node_ids", []), shard.get("texts", [])):
                if node_id in pending:
                    found[node_id] = text
                    pending.remove(node_id)
        return found

    def _lookup_texts(self, node_ids: Sequence[str]) -> Dict[str, str]:
        if self.text_lookup == "shards":
            return self._load_texts_from_shards(node_ids)
        cache = self._load_text_cache()
        return {node_id: cache.get(node_id, "") for node_id in node_ids}

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        k = int(top_k or self.top_k)
        if k <= 0:
            return []
        query_embedding = self._embed_query(query)
        indices, scores = self.strategy.retrieve(query_embedding, self._embeddings, k)
        metadata_by_id = self._load_metadata()
        node_ids = [self._node_ids[idx] for idx in indices]
        texts_by_id = self._lookup_texts(node_ids)
        results = []
        for node_id, score in zip(node_ids, scores):
            results.append(
                RetrievalResult(
                    node_id=node_id,
                    text=texts_by_id.get(node_id, ""),
                    score=float(score),
                    metadata=metadata_by_id.get(node_id, {}),
                )
            )
        return results

    def format_results(self, query: str, top_k: Optional[int] = None) -> str:
        results = self.search(query=query, top_k=top_k)
        if not results:
            return "No results found."
        blocks = []
        for idx, item in enumerate(results, start=1):
            title = item.metadata.get("title") or item.node_id
            blocks.append(
                f"[{idx}] title: {title}\n"
                f"score: {item.score:.4f}\n"
                f"content: {item.text}"
            )
        return "\n\n".join(blocks)
