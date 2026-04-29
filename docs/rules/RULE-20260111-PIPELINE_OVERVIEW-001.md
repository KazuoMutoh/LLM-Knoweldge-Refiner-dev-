# RULE-20260111-PIPELINE_OVERVIEW-001: Rule-Driven KG Refinement Pipeline (v3)

作成日: 2026-01-11
移管元: [docs/pipeline_overview.md](../pipeline_overview.md)

---

This document explains the end-to-end flow implemented by `main_v3.py` using the pipeline interfaces in `simple_active_refine/pipeline.py`.

## High-level stages
1. **Initial load**
   - Read train/valid/test and target triples from a template dataset directory.
   - Instantiate a `RefinedKG` with the initial train triples.
2. **One-shot rule pool build (iteration 0)**
   - `HighScoreRuleExtractor` trains a KGE model, scores triples, mines high-score subgraphs, runs AMIE+, applies exclusion filters, and optionally ranks with `LLMRuleFilter`.
   - The top `n_rules_pool` rules become the fixed pool for all subsequent rounds.
3. **Iterative refinement (iterations 1..N)** via `RuleDrivenKGRefinementPipeline.run`:
   1. **Rule extraction**: `PrecomputedRuleExtractor` simply returns the fixed pool.
   2. **Triple acquisition**: default `RuleBasedTripleAcquirer` (per-rule sampling of target triples, rule body matching over candidate graphs); optionally `RandomTripleAcquirer` if `--use_random_acquirer`.
   3. **Triple evaluation**: `AcceptAllTripleEvaluator` accepts all candidates (placeholder for future scoring/rewards), updates the in-memory KG snapshot.
4. **Final KGE training**
   - `FinalKGETrainer` trains a KGE model on the refined KG and reports Hits@k and MRR.
   - Outputs are written under the working directory.

## Key pipeline interfaces (`simple_active_refine/pipeline.py`)
- `BaseRuleExtractor.extract(ctx) -> RuleExtractionResult`
- `BaseTripleAcquirer.acquire(ctx) -> TripleAcquisitionResult`
- `BaseTripleEvaluator.evaluate(ctx, acquisition) -> TripleEvaluationResult`
- `BaseKGETrainer.train_and_evaluate(ctx) -> KGETrainingResult`
- `RuleDrivenKGRefinementPipeline.run(initial_kg, num_rounds, kge_output_dir)` orchestrates these steps per iteration, then optionally trains KGE.

## Current concrete implementations (v3)
- **Rule extractor (iter 0)**: `HighScoreRuleExtractor`
  - Options: `--use_high_score_triples --lower_percentile --k_neighbor --min_head_coverage --min_pca_conf`
  - LLM filtering: `--use_llm_rule_filter --llm_model --llm_temperature --llm_request_timeout --llm_max_tokens --llm_top_k --llm_min_pca_conf --llm_min_head_coverage`
- **Rule extractor (iter >=1)**: `PrecomputedRuleExtractor` (returns the pool built at iter 0).
- **Triple acquirer**: `RuleBasedTripleAcquirer` (default) or `RandomTripleAcquirer` with `--use_random_acquirer`; both use `--n_targets_per_rule`.
- **Triple evaluator**: `AcceptAllTripleEvaluator` (no filtering yet; writes dumps for inspection).
- **KGE trainer**: `FinalKGETrainer` uses `config_embeddings.json`; epochs overridden by `--num_epochs`.

## Data flow and outputs
- Working directory (`--dir`): per-run sandbox containing iteration folders, tmp artifacts, final datasets, and logs.
- Rule pool diagnostics: `iter_0/rule_extractor_io.json` records extraction parameters, counts, and selected rules.
- Candidate/accepted triples: dumped under the working directory by the acquirer/evaluator implementations.
- Final KGE artifacts and metrics: under `<dir>/final/` and `<dir>/final_dataset`.

## Typical run command (LLM filter on, 1-iter smoke)
```bash
/app/.venv/bin/python main_v3.py \
  --n_iter 1 \
  --num_epochs 1 \
  --dir ./tmp/smoke_llmfilter_on \
  --use_high_score_triples --lower_percentile 80 --k_neighbor 1 \
  --min_head_coverage 0.01 --min_pca_conf 0.05 \
  --use_llm_rule_filter --llm_model gpt-4o --llm_top_k 12
```

## Notes and next steps
- The rule pool is built once and reused; swapping in a smarter evaluator or a different acquirer does not require changing the pipeline contract.
- If LLM scoring returns empty, rules keep a zero score; investigate credentials/rate limits before trusting LLM-based ranking results.
- To harden: add rule-rewarded evaluators, tighter candidate filters, and richer logging/metrics per stage.
