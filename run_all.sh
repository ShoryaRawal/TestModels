#!/usr/bin/env bash
# End-to-end pipeline: data -> tokenizer -> pretraining -> instruction SFT ->
# interpretability report.
#
# Runs under `caffeinate -i` so the Mac does not idle-sleep mid-training.
# Everything is teed to training.log.
set -euo pipefail

# Re-exec under caffeinate exactly once.
if [[ "${_TINY_PYTHIA_CAFFEINATED:-0}" != "1" ]]; then
  export _TINY_PYTHIA_CAFFEINATED=1
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  exec caffeinate -i "$SELF" "$@"
fi

cd "$(dirname "$0")"

LOG="training.log"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG") 2>&1

START_TS=$(date +%s)
echo "======================================================================"
echo " tiny-pythia pipeline"
echo " started: $(date '+%Y-%m-%d %H:%M:%S')"
echo " host:    $(hostname)  $(uname -m)"
echo " log:     $(pwd)/$LOG"
echo "======================================================================"

fail() {
  echo ""
  echo "FAILED at $(date '+%Y-%m-%d %H:%M:%S')"
  exit 1
}
trap fail ERR

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [[ ! -x ".venv/bin/python" ]]; then
  echo "[setup] creating .venv"
  python3 -m venv .venv
fi
PY=".venv/bin/python"

echo "[setup] installing requirements"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt

"$PY" - <<'PYCHECK'
import pathlib, re
from importlib.metadata import version
print(f"[setup] mlx={version('mlx')} numpy={version('numpy')} "
      f"tokenizers={version('tokenizers')} safetensors={version('safetensors')}")
forbidden = ("transformers", "datasets", "huggingface_hub", "accelerate")
hits = []
for py in pathlib.Path(".").glob("*.py"):
    src = py.read_text(encoding="utf-8", errors="ignore")
    for mod in forbidden:
        if re.search(rf"^\s*(?:import|from)\s+{mod}\b", src, flags=re.MULTILINE):
            hits.append(f"{py}:{mod}")
if hits:
    raise SystemExit("[setup] forbidden Hugging Face imports: " + ", ".join(hits))
print("[setup] no Hugging Face imports in project sources")
PYCHECK

mkdir -p data artifacts checkpoints/pretrain checkpoints/instruct

run_step() {
  local label="$1"; shift
  local t0 t1
  t0=$(date +%s)
  echo ""
  echo "----------------------------------------------------------------------"
  echo "[$label] started $(date '+%H:%M:%S')"
  echo "----------------------------------------------------------------------"
  "$@"
  t1=$(date +%s)
  echo "[$label] finished in $((t1 - t0))s"
}

# ---------------------------------------------------------------------------
# Pipeline (Phase 1 ~26 min, Phase 2 ~5 min on an M1 16GB; calibrated live)
# ---------------------------------------------------------------------------
run_step "1/6 fetch + synthesize data" \
  "$PY" 01_fetch_data.py \
    --train-mb 150 \
    --num-instruct 15000 \
    --num-valid-instruct 500

run_step "2/6 tokenizer + pre-tokenize" \
  "$PY" 02_train_tokenizer.py \
    --vocab-size 8192 \
    --tokenizer-sample-mb 60

run_step "3/6 base pretraining" \
  "$PY" 03_pretrain.py \
    --minutes 26 \
    --seq-len 256 \
    --batch-size 32 \
    --lr 6e-4

run_step "4/7 build mixed SFT data" \
  "$PY" build_sft_data.py \
    --conversational 5000 \
    --valid-conversational 400

run_step "5/7 instruction SFT" \
  "$PY" 04_instruct_sft.py \
    --minutes 5 \
    --seq-len 1024 \
    --max-seq-len 1024 \
    --batch-size 8 \
    --lr 1.5e-4

run_step "6/7 RLVR learning loop (GRPO)" \
  "$PY" 06_rl_loop.py \
    --rounds 4 \
    --steps-per-round 25

run_step "7/7 interpretability lab" \
  "$PY" 05_interpretability_lab.py

END_TS=$(date +%s)
echo ""
echo "======================================================================"
echo " COMPLETE in $(((END_TS - START_TS) / 60))m $(((END_TS - START_TS) % 60))s"
echo " finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
echo ""
echo "Artifacts:"
ls -lh artifacts/ 2>/dev/null || true
echo ""
echo "Checkpoints:"
find checkpoints -name '*.safetensors' -maxdepth 2 -print0 2>/dev/null \
  | xargs -0 ls -lh 2>/dev/null | tail -25 || true
echo ""
echo "Final metrics:"
tail -n 3 artifacts/pretrain_metrics.jsonl 2>/dev/null || true
tail -n 3 artifacts/sft_metrics.jsonl 2>/dev/null || true
echo ""
echo "Check instruction behavior:"
echo "  .venv/bin/python test_instruct.py"
echo "Chat with the instruction model:"
echo "  .venv/bin/python 05_interpretability_lab.py --chat"
echo "Full report:"
echo "  artifacts/interpretability_report.txt"
echo "======================================================================"
