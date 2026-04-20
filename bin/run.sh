#!/usr/bin/env bash
set -euo pipefail

# ---- paths ----
REPO_DIR="/app/spoken-numbers-recognition"
DATA_DIR="${REPO_DIR}/data/processed_16k"
ART_DIR="${REPO_DIR}/artifacts"
RUN_DIR="${ART_DIR}/single_best"

cd "${REPO_DIR}"

echo "[run.sh] python: $(python --version 2>&1)"
echo "[run.sh] nvidia-smi:"
nvidia-smi || true

# ---- install uv ----
if ! command -v uv >/dev/null 2>&1; then
  echo "[run.sh] installing uv..."
  pip install --no-cache-dir uv
fi

# ---- deps ----
echo "[run.sh] uv sync..."
export UV_PROJECT_ENVIRONMENT="${REPO_DIR}/.venv"
uv sync --frozen --no-dev

echo "[run.sh] data root contents:"
ls -la "${DATA_DIR}" || { echo "[run.sh] data missing at ${DATA_DIR}"; exit 1; }

mkdir -p "${ART_DIR}" "${RUN_DIR}"

MODE="${RUN_MODE:-train_and_submit}"

# ---- single-run env defaults (used only when MODE=train / train_and_submit) ----
# Tuned for the "augmented" regime: SpecAugment + speed perturbation + cosine LR.
# With strong augmentation we can train longer at higher peak LR without overfitting.
DEVICE="${DEVICE:-cuda}"
TOKENIZER="${TOKENIZER:-russian_number_compact}"
N_MELS="${N_MELS:-80}"
ENCODER_DIM="${ENCODER_DIM:-224}"
ENCODER_LAYERS="${ENCODER_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-8e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-20}"
EPOCHS="${EPOCHS:-120}"
SEED="${SEED:-42}"
# Augmentation / scheduler
AUGMENT="${AUGMENT:-1}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-4}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.02}"
AUG_FREQ_MASK_NUM="${AUG_FREQ_MASK_NUM:-2}"
AUG_FREQ_MASK_WIDTH="${AUG_FREQ_MASK_WIDTH:-15}"
AUG_TIME_MASK_NUM="${AUG_TIME_MASK_NUM:-2}"
AUG_TIME_MASK_WIDTH="${AUG_TIME_MASK_WIDTH:-40}"
AUG_TIME_MASK_RATIO="${AUG_TIME_MASK_RATIO:-0.25}"
AUG_SPEED_PROB="${AUG_SPEED_PROB:-0.6}"
AUG_SPEED_MIN="${AUG_SPEED_MIN:-0.9}"
AUG_SPEED_MAX="${AUG_SPEED_MAX:-1.1}"

# Build a synthetic sample_submission.csv from test.csv if the real one
# is missing (make_submission.py requires it; only length + original-ext
# filenames are needed for proper reindex).
ensure_sample_submission() {
  local real="${DATA_DIR}/sample_submission.csv"
  if [[ -f "${real}" ]]; then
    echo "${real}"
    return 0
  fi
  local synth="${ART_DIR}/sample_submission_synth.csv"
  if [[ ! -f "${synth}" ]]; then
    uv run python - <<PY > /dev/null
from pathlib import Path
import pandas as pd

df = pd.read_csv("${DATA_DIR}/test.csv")
def restore(row):
    ext = str(row["ext"]).lstrip(".")
    return Path(str(row["filename"])).with_suffix(f".{ext}").as_posix()

out = pd.DataFrame({
    "filename": df.apply(restore, axis=1),
    "transcription": ["0"] * len(df),
})
out.to_csv("${synth}", index=False)
PY
  fi
  echo "${synth}"
}

# Train one config and build its submission.
# Usage: run_variant <name> <train cli args...>
run_variant() {
  local name="$1"; shift
  local run_dir="${ART_DIR}/variant_${name}"
  mkdir -p "${run_dir}"
  echo ""
  echo "[variants] =================================================="
  echo "[variants] starting ${name} -> ${run_dir}"
  echo "[variants] =================================================="
  uv run python -m src.train_baseline \
    --data-root "${DATA_DIR}" \
    --output-dir "${run_dir}" \
    --device cuda \
    --n-mels 80 \
    --num-workers "${NUM_WORKERS}" \
    --save-all-checkpoints \
    "$@"

  if [[ ! -f "${run_dir}/best.pt" ]]; then
    echo "[variants] WARN: ${run_dir}/best.pt missing, skipping submission" >&2
    return 0
  fi

  local sample
  sample="$(ensure_sample_submission)"
  echo "[variants] building submission for ${name}"
  uv run python -m src.make_submission \
    --checkpoint "${run_dir}/best.pt" \
    --data-root "${DATA_DIR}" \
    --sample-submission "${sample}" \
    --output "${run_dir}/submission.csv" \
    --device cuda \
    --batch-size 512 \
    --num-workers 8
}

run_train() {
  echo "[run.sh] launching single train run -> ${RUN_DIR}"
  local extra=()
  if [[ "${AUGMENT}" == "1" || "${AUGMENT,,}" == "true" ]]; then
    extra+=(
      --augment
      --aug-speed-prob "${AUG_SPEED_PROB}"
      --aug-speed-min "${AUG_SPEED_MIN}"
      --aug-speed-max "${AUG_SPEED_MAX}"
      --aug-freq-mask-num "${AUG_FREQ_MASK_NUM}"
      --aug-freq-mask-width "${AUG_FREQ_MASK_WIDTH}"
      --aug-time-mask-num "${AUG_TIME_MASK_NUM}"
      --aug-time-mask-width "${AUG_TIME_MASK_WIDTH}"
      --aug-time-mask-ratio "${AUG_TIME_MASK_RATIO}"
    )
  fi
  if [[ "${SCHEDULER}" != "none" ]]; then
    extra+=(
      --scheduler "${SCHEDULER}"
      --warmup-epochs "${WARMUP_EPOCHS}"
      --min-lr-ratio "${MIN_LR_RATIO}"
    )
  fi

  uv run python -m src.train_baseline \
    --data-root "${DATA_DIR}" \
    --output-dir "${RUN_DIR}" \
    --device "${DEVICE}" \
    --tokenizer "${TOKENIZER}" \
    --n-mels "${N_MELS}" \
    --encoder-dim "${ENCODER_DIM}" \
    --encoder-layers "${ENCODER_LAYERS}" \
    --dropout "${DROPOUT}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --save-all-checkpoints \
    "${extra[@]}"
}

run_submit() {
  local ckpt="${CHECKPOINT:-${RUN_DIR}/best.pt}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[run.sh] checkpoint missing: ${ckpt}" >&2
    exit 3
  fi
  local sample
  sample="$(ensure_sample_submission)"
  echo "[run.sh] building submission from ${ckpt}"
  uv run python -m src.make_submission \
    --checkpoint "${ckpt}" \
    --data-root "${DATA_DIR}" \
    --sample-submission "${sample}" \
    --output "${RUN_DIR}/submission.csv" \
    --device "${DEVICE}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
}

# --------- the 4 diverse variants ---------
# Each tests a distinct hypothesis about what helps CER on this task.
run_multi() {
  # V1: "deep compact" — главная гипотеза.
  # Compact tokenizer + 3 слоя BiGRU шириной 224 (≈3.85M) + умеренный dropout.
  # Гипотеза: короткие таргеты + глубина лучше всего ловят OOD.
  run_variant "V1_compact_deep" \
    --tokenizer russian_number_compact \
    --encoder-dim 224 --encoder-layers 3 \
    --dropout 0.15 \
    --learning-rate 5e-4 --weight-decay 1e-2 \
    --batch-size 256 --eval-batch-size 512 \
    --epochs 45 --seed 42

  # V2: "wide compact" — максимальная ширина под лимит 5M.
  # Compact tokenizer + 2 слоя × 320 (≈4.94M).
  # Гипотеза: ширина даёт больше "памяти" классам, при малой глубине легче учить.
  run_variant "V2_compact_wide" \
    --tokenizer russian_number_compact \
    --encoder-dim 320 --encoder-layers 2 \
    --dropout 0.15 \
    --learning-rate 5e-4 --weight-decay 1e-2 \
    --batch-size 256 --eval-batch-size 512 \
    --epochs 35 --seed 42

  # V3: "words deep reference" — проверка, нужна ли вообще compact.
  # Words tokenizer + 3 слоя × 224 (≈3.84M).
  # Гипотеза: семантически богаче, за счёт длины может ловить ошибки чисел лучше.
  run_variant "V3_words_deep" \
    --tokenizer russian_number_words \
    --encoder-dim 224 --encoder-layers 3 \
    --dropout 0.15 \
    --learning-rate 3e-4 --weight-decay 1e-2 \
    --batch-size 256 --eval-batch-size 512 \
    --epochs 45 --seed 42

  # V4: "small-batch heavy-reg" — противоположность big-batch режиму.
  # Compact, 3×224, dropout=0.25, batch 64, low LR, 50 эпох.
  # Гипотеза: много стохастичных шагов + сильная регуляризация = лучшая генерализация OOD.
  run_variant "V4_compact_small_batch_reg" \
    --tokenizer russian_number_compact \
    --encoder-dim 224 --encoder-layers 3 \
    --dropout 0.25 \
    --learning-rate 3e-4 --weight-decay 5e-2 \
    --batch-size 64 --eval-batch-size 256 \
    --epochs 50 --seed 42
}

case "${MODE}" in
  multi_train_and_submit)
    run_multi
    ;;
  train_and_submit)
    run_train
    run_submit
    ;;
  train)
    run_train
    ;;
  submit)
    run_submit
    ;;
  sweep)
    echo "[run.sh] launching sweep (configs/sweep_baseline.yaml)"
    uv run python -m src.sweep_baseline \
      --config "${SWEEP_CONFIG:-configs/sweep_baseline.yaml}"
    ;;
  *)
    echo "[run.sh] unknown RUN_MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo ""
echo "[run.sh] done. artifacts tree:"
find "${ART_DIR}" -maxdepth 3 -type f | sort | head -n 60

# Quick summary of best dev CER per variant (if variants ran).
echo ""
echo "[run.sh] per-variant best dev metrics (from history.json):"
for hjson in "${ART_DIR}"/variant_*/history.json; do
  [[ -f "${hjson}" ]] || continue
  uv run python - "${hjson}" <<'PY' || true
import json, sys
path = sys.argv[1]
with open(path) as f:
    hist = json.load(f)
if not hist:
    print(f"{path}: empty")
    raise SystemExit
def pick(ep):
    return ep.get("dev_primary_hmean_cer") if ep.get("dev_primary_hmean_cer") is not None else ep.get("dev_cer")
best = min(hist, key=lambda e: (pick(e) if pick(e) is not None else 1e9))
name = path.split("/variant_")[1].split("/")[0]
print(f"  {name:35s} epoch={best['epoch']:3d}  "
      f"dev_primary_hmean_cer={best.get('dev_primary_hmean_cer')}  "
      f"dev_cer={best.get('dev_cer')}  "
      f"in_domain={best.get('dev_in_domain_cer')}  "
      f"ood={best.get('dev_out_of_domain_cer')}")
PY
done
