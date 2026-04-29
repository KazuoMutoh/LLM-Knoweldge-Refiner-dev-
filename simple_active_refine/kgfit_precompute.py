from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from simple_active_refine.util import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class KGFitPrecomputeConfig:
    model: str = "text-embedding-3-small"
    batch_size: int = 128
    dtype: str = "float32"
    use_name_as_desc_if_missing: bool = True


def read_entity_texts(path: Path) -> Dict[str, str]:
    """Read tab-separated entity -> text mapping."""
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                continue
            entity, text = line.split("\t", 1)
            mapping[entity] = text
    return mapping


def read_entities_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_entity_order(*, dir_triples: Path, name_texts: Dict[str, str]) -> List[str]:
    entities_path = dir_triples / "entities.txt"
    entities = read_entities_file(entities_path)
    if entities:
        return entities
    return sorted(name_texts.keys())


def _ensure_complete_texts(
    *,
    entities: Sequence[str],
    name_texts: Dict[str, str],
    desc_texts: Dict[str, str],
    use_name_as_desc_if_missing: bool,
) -> Tuple[List[str], List[str]]:
    name_list: List[str] = []
    desc_list: List[str] = []
    missing = []
    for e in entities:
        name = name_texts.get(e)
        desc = desc_texts.get(e)
        if name is None:
            missing.append(e)
            name = e
        if desc is None and use_name_as_desc_if_missing:
            desc = name
        if desc is None:
            desc = ""
        name_list.append(name)
        desc_list.append(desc)
    if missing:
        logger.warning("Missing name text for %d entities. Example: %s", len(missing), missing[:3])
    return name_list, desc_list


def embed_texts(
    *,
    texts: Sequence[str],
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    batch_size: int,
) -> np.ndarray:
    vectors: List[Sequence[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(embed_fn(batch))
    return np.asarray(vectors)


def save_kgfit_embeddings(
    *,
    output_dir: Path,
    entity_to_row: Dict[str, int],
    name_embeddings: np.ndarray,
    desc_embeddings: np.ndarray,
    model: str,
    name_source: Path,
    desc_source: Path,
    dtype: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name_path = output_dir / "entity_name_embeddings.npy"
    desc_path = output_dir / "entity_desc_embeddings.npy"
    meta_path = output_dir / "entity_embedding_meta.json"

    np.save(name_path, name_embeddings)
    np.save(desc_path, desc_embeddings)

    meta = {
        "provider": "openai",
        "model": model,
        "dim": int(name_embeddings.shape[1]),
        "dtype": dtype,
        "entity_to_row": entity_to_row,
        "text_sources": [str(name_source), str(desc_source)],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved name embeddings: %s", name_path)
    logger.info("Saved desc embeddings: %s", desc_path)
    logger.info("Saved meta: %s", meta_path)


def build_embedder(model: str) -> Callable[[Sequence[str]], Sequence[Sequence[float]]]:
    if "OPENAI_API_KEY" not in (dict(**__import__("os").environ)):
        try:
            from settings import OPENAI_API_KEY  # type: ignore

            if OPENAI_API_KEY:
                __import__("os").environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        except Exception:
            pass
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI()

        def _embed(batch: Sequence[str]) -> Sequence[Sequence[float]]:
            resp = client.embeddings.create(model=model, input=list(batch))
            return [item.embedding for item in resp.data]

        return _embed
    except Exception:
        from langchain_openai import OpenAIEmbeddings  # type: ignore

        embedder = OpenAIEmbeddings(model=model)

        def _embed(batch: Sequence[str]) -> Sequence[Sequence[float]]:
            return embedder.embed_documents(list(batch))

        return _embed


def precompute_kgfit_embeddings(
    *,
    dir_triples: Path,
    output_dir: Optional[Path] = None,
    name_source: str = "entity2text.txt",
    desc_source: str = "entity2textlong.txt",
    config: Optional[KGFitPrecomputeConfig] = None,
    max_items: Optional[int] = None,
) -> None:
    config = config or KGFitPrecomputeConfig()
    output_dir = output_dir or (dir_triples / ".cache" / "kgfit")

    name_path = dir_triples / name_source
    desc_path = dir_triples / desc_source

    name_texts = read_entity_texts(name_path)
    desc_texts = read_entity_texts(desc_path)
    entities = resolve_entity_order(dir_triples=dir_triples, name_texts=name_texts)
    if max_items is not None:
        entities = entities[:max_items]

    name_list, desc_list = _ensure_complete_texts(
        entities=entities,
        name_texts=name_texts,
        desc_texts=desc_texts,
        use_name_as_desc_if_missing=config.use_name_as_desc_if_missing,
    )

    embed_fn = build_embedder(config.model)
    name_emb = embed_texts(texts=name_list, embed_fn=embed_fn, batch_size=config.batch_size)
    desc_emb = embed_texts(texts=desc_list, embed_fn=embed_fn, batch_size=config.batch_size)

    if config.dtype == "float16":
        name_emb = name_emb.astype(np.float16)
        desc_emb = desc_emb.astype(np.float16)
    else:
        name_emb = name_emb.astype(np.float32)
        desc_emb = desc_emb.astype(np.float32)

    entity_to_row = {entity: idx for idx, entity in enumerate(entities)}
    save_kgfit_embeddings(
        output_dir=output_dir,
        entity_to_row=entity_to_row,
        name_embeddings=name_emb,
        desc_embeddings=desc_emb,
        model=config.model,
        name_source=name_path,
        desc_source=desc_path,
        dtype=config.dtype,
    )
