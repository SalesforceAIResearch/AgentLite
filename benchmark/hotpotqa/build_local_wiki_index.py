import argparse
import bz2
import json
import re
import shutil
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


DEFAULT_INPUT_DIR = "/home/xuzm/data/ICCAD2026/rag_cag_algorithm/TurboRAG/wikpedia_paragraphs"
DEFAULT_PERSIST_DIR = "/home/xuzm/data/ICCAD2026/rag_cag_algorithm/TurboRAG/emb_wiki_bz2_nomic_agentlite"
DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_EMBEDDING_DIM = 256


def parse_args():
    parser = argparse.ArgumentParser(description="Build a simple local Wikipedia RAG index.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--persist_dir", type=str, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--embed_model_name", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--encode_batch_size", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_bz_files(input_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.name.lower().endswith((".bz2", ".bz"))
    ]


def iter_jsonl_records(path: Path) -> Iterator[Tuple[int, Optional[dict]]]:
    with bz2.open(path, mode="rt", encoding="utf-8", errors="replace") as fin:
        for line_no, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def normalize_text_segments(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(part).strip() for part in value if str(part).strip()).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def sanitize_doc_key(raw_value: object, fallback_index: int) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return f"doc{fallback_index}"
    text = re.sub(r"[^0-9A-Za-z-]+", "-", text).strip("-")
    return text or f"doc{fallback_index}"


def sanitize_relpath_tag(path: str) -> str:
    return re.sub(r"[^0-9A-Za-z-]+", "-", path).strip("-") or "src"


def build_node_text(title: str, body: str) -> str:
    title = title.strip()
    body = body.strip()
    if title and body:
        return f"Title: {title}\n\n{body}"
    return title or body


def build_record_payload(
    record: dict,
    source_file: Path,
    source_relpath: str,
    source_line: int,
    fallback_index: int,
):
    title = str(record.get("title") or "").strip()
    body = normalize_text_segments(record.get("text"))
    node_text = build_node_text(title, body)
    if not node_text:
        return None, None, None

    doc_key = sanitize_doc_key(record.get("id"), fallback_index)
    source_tag = sanitize_relpath_tag(source_relpath)
    node_id = f"chunk_{doc_key}_{source_tag}-{source_line}"
    metadata = {
        "node_id": node_id,
        "record_id": str(record.get("id") or ""),
        "title": title,
        "url": str(record.get("url") or ""),
        "source_file": source_file.name,
        "source_relpath": source_relpath,
        "source_line": source_line,
    }
    return node_id, node_text, metadata


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        cache_dir: Optional[str] = None,
        encode_batch_size: Optional[int] = None,
    ) -> None:
        import torch
        import torch.nn.functional as F
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.F = F
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.encode_batch_size = encode_batch_size
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            cache_folder=cache_dir,
        )
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = 8192

    def encode_documents(self, texts: List[str]):
        normalized_name = self.model_name.lower()
        encode_kwargs = {"convert_to_tensor": True, "show_progress_bar": False}
        if self.encode_batch_size is not None:
            encode_kwargs["batch_size"] = self.encode_batch_size
        if "nomic-embed-text-v1.5" in normalized_name:
            embeddings = self.model.encode(
                [f"search_document: {text}" for text in texts],
                **encode_kwargs,
            )
            embeddings = embeddings.to(self.torch.float32)
            embeddings = self.F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, : self.embedding_dim]
            embeddings = self.F.normalize(embeddings, p=2, dim=1)
            return embeddings.to(self.torch.float16)
        if "jina-embeddings-v3" in normalized_name:
            embeddings = self.model.encode(
                texts,
                task="retrieval.passage",
                prompt_name="retrieval.passage",
                truncate_dim=self.embedding_dim,
                **encode_kwargs,
            )
            embeddings = embeddings.to(self.torch.float32)
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, : self.embedding_dim]
            embeddings = self.F.normalize(embeddings, p=2, dim=1)
            return embeddings.to(self.torch.float16)
        raise ValueError(f"Unsupported embedding model: {self.model_name}")


def shard_dir(persist_dir: Path) -> Path:
    return persist_dir / "shards"


def save_shard(persist_dir: Path, shard_idx: int, node_ids, texts, metadata, embs) -> None:
    import torch

    path = shard_dir(persist_dir) / f"shard_{shard_idx:06d}.pt"
    torch.save(
        {"node_ids": node_ids, "texts": texts, "metadata": metadata, "embs": embs},
        path,
    )


def finalize_outputs(persist_dir: Path, stats: dict) -> None:
    import torch

    all_node_ids = []
    embedding_chunks = []
    text_cache = {}
    metadata_path = persist_dir / "codex_node_metadata.jsonl"

    with open(metadata_path, "w", encoding="utf-8") as metadata_out:
        for shard_path in sorted(shard_dir(persist_dir).glob("shard_*.pt")):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            node_ids = shard["node_ids"]
            texts = shard["texts"]
            metadata = shard["metadata"]
            embs = shard["embs"]
            all_node_ids.extend(node_ids)
            embedding_chunks.append(embs.to(torch.float16))
            for node_id, text in zip(node_ids, texts):
                text_cache[node_id] = text
            for meta in metadata:
                metadata_out.write(json.dumps(meta, ensure_ascii=False) + "\n")

    torch.save(
        {"node_ids": all_node_ids, "embs": torch.cat(embedding_chunks, dim=0)},
        persist_dir / "codex_vector_cache_float16.pt",
    )
    torch.save(text_cache, persist_dir / "codex_node_text_cache.pt")
    with open(persist_dir / "codex_build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    persist_dir = Path(args.persist_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if persist_dir.exists():
        if not args.overwrite:
            raise ValueError(f"Persist dir exists; pass --overwrite to replace it: {persist_dir}")
        shutil.rmtree(persist_dir)
    shard_dir(persist_dir).mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformerEmbedder(
        model_name=args.embed_model_name,
        embedding_dim=args.embedding_dim,
        cache_dir=args.cache_dir,
        encode_batch_size=args.encode_batch_size or args.batch_size,
    )
    bz_files = iter_bz_files(input_dir)
    if args.max_files is not None:
        bz_files = bz_files[: args.max_files]

    batch_node_ids, batch_texts, batch_metadata = [], [], []
    inserted_records = skipped_records = malformed_records = total_chars = shard_idx = 0
    start_time = time.time()

    for file_idx, bz_path in enumerate(bz_files, start=1):
        source_relpath = bz_path.relative_to(input_dir).as_posix()
        print(f"[{file_idx}/{len(bz_files)}] Reading {source_relpath}")
        for line_no, record in iter_jsonl_records(bz_path):
            if args.max_records is not None and inserted_records >= args.max_records:
                break
            if record is None:
                malformed_records += 1
                continue
            node_id, text, metadata = build_record_payload(
                record=record,
                source_file=bz_path,
                source_relpath=source_relpath,
                source_line=line_no,
                fallback_index=inserted_records + skipped_records + malformed_records,
            )
            if node_id is None:
                skipped_records += 1
                continue
            batch_node_ids.append(node_id)
            batch_texts.append(text)
            batch_metadata.append(metadata)
            if len(batch_node_ids) >= args.batch_size:
                embs = embedder.encode_documents(batch_texts)
                save_shard(persist_dir, shard_idx, batch_node_ids, batch_texts, batch_metadata, embs)
                shard_idx += 1
                inserted_records += len(batch_node_ids)
                total_chars += sum(len(item) for item in batch_texts)
                batch_node_ids, batch_texts, batch_metadata = [], [], []
                if args.log_every > 0 and inserted_records % args.log_every == 0:
                    elapsed = max(time.time() - start_time, 1e-6)
                    print(f"Inserted {inserted_records} records ({inserted_records / elapsed:.2f}/sec)")
        if args.max_records is not None and inserted_records >= args.max_records:
            break

    if batch_node_ids:
        embs = embedder.encode_documents(batch_texts)
        save_shard(persist_dir, shard_idx, batch_node_ids, batch_texts, batch_metadata, embs)
        inserted_records += len(batch_node_ids)
        total_chars += sum(len(item) for item in batch_texts)
        shard_idx += 1

    stats = {
        "inserted_records": inserted_records,
        "skipped_records": skipped_records,
        "malformed_records": malformed_records,
        "total_chars": total_chars,
        "avg_chars": total_chars / max(inserted_records, 1),
        "embed_model_name": args.embed_model_name,
        "embedding_dim": args.embedding_dim,
        "num_shards": shard_idx,
        "persist_mode": "agentlite_local_wiki_simple",
        "finalized_at": time.time(),
    }
    finalize_outputs(persist_dir, stats)
    print("Done.")
    print(f"Index dir: {persist_dir}")
    print(f"Inserted records: {inserted_records}")


if __name__ == "__main__":
    main()
