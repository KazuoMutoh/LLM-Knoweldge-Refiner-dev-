#!/usr/bin/env bash
set -euo pipefail

# Runs remaining relations sequentially for rerun1:
#   1) place_of_birth: UCB vs Random(seed=0) + report
#   2) profession: train before model, then UCB vs Random(seed=0) + report
#
# This script is GPU-heavy; it is intentionally sequential.

if [ -f /app/.venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source /app/.venv/bin/activate
fi

BASE_OUT_DIR=/app/experiments/20260126_rerun1
MODELS_DIR=/app/models/20260126_rerun1
EMBED_CFG=/app/config_embeddings_kgfit_pairre_fb15k237_lowmem_bs128.json
EPOCHS=100

mkdir -p "$BASE_OUT_DIR" "$MODELS_DIR"

run_relation() {
  local relation_name="$1"
  local target_relation="$2"
  local dataset_dir="$3"
  local run_suffix="$4"
  local model_before_dir="$5"

  python3 /app/scripts/run_ucb_vs_random_seed0_and_report.py \
    --base_out_dir "$BASE_OUT_DIR" \
    --run_suffix "$run_suffix" \
    --relation_name "$relation_name" \
    --target_relation "$target_relation" \
    --dataset_dir "$dataset_dir" \
    --target_triples "$dataset_dir/target_triples.txt" \
    --candidate_triples "$dataset_dir/train_removed.txt" \
    --model_before_dir "$model_before_dir" \
    --embedding_config "$EMBED_CFG" \
    --num_epochs "$EPOCHS"
}

train_before_if_missing() {
  local dataset_dir="$1"
  local out_dir="$2"

  if [ -f "$out_dir/trained_model.pkl" ]; then
    echo "[before] exists: $out_dir"
    return 0
  fi

  mkdir -p "$out_dir"
  echo "[before] training: $out_dir"

  # Note: --skip_evaluation is intentional (plan spec)
  python3 /app/scripts/train_initial_kge.py \
    --dir_triples "$dataset_dir" \
    --output_dir "$out_dir" \
    --embedding_config "$EMBED_CFG" \
    --num_epochs "$EPOCHS" \
    --skip_evaluation \
    2>&1 | tee "$out_dir/train.log"
}

# -----------------------------------------------------------------------------
# 1) place_of_birth
# -----------------------------------------------------------------------------
POB_DATASET=/app/experiments/test_data_for_place_of_birth_head_incident_v1_kgfit
POB_BEFORE="$MODELS_DIR/fb15k237_kgfit_pairre_place_of_birth_head_incident_v1_before_ep100_lowmem_bs128"

if [ ! -f "$POB_BEFORE/trained_model.pkl" ]; then
  echo "[error] place_of_birth before model not found: $POB_BEFORE"
  exit 2
fi

echo "[run] place_of_birth UCB vs Random(seed0)"
run_relation \
  "place_of_birth" \
  "/people/person/place_of_birth" \
  "$POB_DATASET" \
  "20260127a" \
  "$POB_BEFORE"

# -----------------------------------------------------------------------------
# 2) profession
# -----------------------------------------------------------------------------
PROF_DATASET=/app/experiments/test_data_for_profession_head_incident_v1_kgfit
PROF_BEFORE="$MODELS_DIR/fb15k237_kgfit_pairre_profession_head_incident_v1_before_ep100_lowmem_bs128"

train_before_if_missing "$PROF_DATASET" "$PROF_BEFORE"

echo "[run] profession UCB vs Random(seed0)"
run_relation \
  "profession" \
  "/people/person/profession" \
  "$PROF_DATASET" \
  "20260127b" \
  "$PROF_BEFORE"

echo "[done] all remaining runs finished"
