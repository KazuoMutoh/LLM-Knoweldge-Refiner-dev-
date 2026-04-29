# RULE-20260111-PIPELINE_CONCRETE-001: Pipeline concrete implementations

作成日: 2026-01-11
移管元: [docs/pipeline_concrete.md](../pipeline_concrete.md)

---

This note summarizes the concrete strategy classes added under `simple_active_refine/pipeline_concrete.py`.

## Rule extractors
- `AMIERuleExtractor`: runs AMIE+ with numeric filters (`min_pca_conf`, `min_head_coverage`, optional `max_body_len`).
- `AMIERuleExtractorWithLLMFilter`: same as above, then applies an injected `llm_filter(rule) -> bool` callable. Useful for LLM ranking or mocking.

## Triple acquirers
- `TrainRemovedTripleAcquirer`: reads `train_removed.txt` (path provided via `metadata["train_removed_path"]`), deduplicates against the current KG, and returns candidates under the key `train_removed`.
- `LLMWebTripleAcquirer`: thin wrapper around `LLMKnowledgeRetriever`; accepts an injected `retrieval_fn(retriever, rule, triples)` for custom Web/LLM retrieval. If no retriever is provided, returns empty candidates.

## Triple evaluator
- `SimpleHeuristicTripleEvaluator`: optional entity-text gating via `metadata["entity_text_path"]`; accepts triples whose head/tail appear in the text file, assigns score 1.0, and produces simple rule rewards. Designed to be subclassed with richer scoring.

## KGE trainer
- `PyKEENKGETrainer`: trains and evaluates a PyKEEN model (`model_name`, `num_epochs`, `create_inverse_triples`, `pipeline_kwargs`). Requires `metadata["dir_triples"]` and writes to `output_dir` if provided.

## Orchestration
Use these classes with `RuleDrivenKGRefinementPipeline` (in `pipeline.py`). Example:
```python
from simple_active_refine.pipeline import RuleDrivenKGRefinementPipeline, RefinedKG
from simple_active_refine.pipeline_concrete import (
    AMIERuleExtractor, TrainRemovedTripleAcquirer,
    SimpleHeuristicTripleEvaluator, PyKEENKGETrainer,
)

pipeline = RuleDrivenKGRefinementPipeline(
    rule_extractor=AMIERuleExtractor(),
    triple_acquirer=TrainRemovedTripleAcquirer(),
    triple_evaluator=SimpleHeuristicTripleEvaluator(),
    kge_trainer=PyKEENKGETrainer(num_epochs=1),
)
result = pipeline.run(
    initial_kg=RefinedKG(triples=[("h1","r1","t1")]),
    num_rounds=1,
    kge_output_dir="./tmp/kge_run",
)
```

## Notes
- Interfaces are stable; implementations can be subclassed or swapped without changing pipeline orchestration.
- For heavy dependencies (AMIE+, LLM), inject callables or mock them in tests to keep runs lightweight.
