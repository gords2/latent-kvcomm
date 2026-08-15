#!/usr/bin/env bash
# Operational tooling, NOT a plan deliverable: runs the remaining gates back to
# back so a billing GPU box is never idle between them.
#
# Ordering note. The plan says to evaluate each gate before the next one "means
# anything", and that remains true for INTERPRETATION -- L2 tells you nothing
# about the relay if L1 failed. It is not a reason to leave the GPU idle: the
# runs are mechanically independent, the box bills either way, and a failed
# gate's data is exactly what its diagnostic ladder needs. So this COLLECTS
# everything and EVALUATES nothing. A human reads the tables, in order.
#
# Every gate script skips conditions whose JSON already exists, so this is safe
# to re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
mkdir -p runs

echo "### waiting for any in-flight gate_latentmas.sh"
while pgrep -f "gate_latentmas.sh" >/dev/null; do sleep 60; done
echo "### L1 sweep finished at $(date -u +%FT%TZ)"

# S_BEST: the latent_steps with the highest accuracy in the L1 sweep. Task 11
# uses it for MBPP+ and for the 4B/14B runs. Falls back to 40 (mid-range of the
# paper's [0,80]) only if no L1 record can be read, so the chain still produces
# something rather than dying on a missing file.
S_BEST=$($PY - <<'EOF'
import glob, json
best, acc = None, -1.0
for p in glob.glob("runs/lm_gsm8k_8b_s*.json"):
    d = json.load(open(p))
    if d["system"] == "latentmas" and d["accuracy"] > acc:
        best, acc = d["latent_steps"], d["accuracy"]
print(best if best is not None else 40)
EOF
)
echo "### S_BEST = $S_BEST"

run() {  # run <tag> <args...>
  local tag=$1; shift
  local f="runs/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag (exists)"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --seed 42 --generate_bs 16 --out_json "$f" "$@" 2>&1 | tail -15
}

echo
echo "############ Task 11 -- L2"
run lm_mbppplus_8b --system latentmas --task mbppplus \
    --model_name Qwen/Qwen3-8B --max_samples 378 --latent_steps "$S_BEST"
run lm_mbppplus_8b_single --system single --task mbppplus \
    --model_name Qwen/Qwen3-8B --max_samples 378 --latent_steps 0
run lm_gsm8k_4b --system latentmas --task gsm8k \
    --model_name Qwen/Qwen3-4B --max_samples 200 --latent_steps "$S_BEST"
run lm_gsm8k_14b --system latentmas --task gsm8k \
    --model_name Qwen/Qwen3-14B --max_samples 200 --latent_steps "$S_BEST"

echo
echo "############ Task 15 -- K1 (countries, ungated Llama mirror)"
TASK=countries ./scripts/gate_kvcomm.sh

echo
echo "############ Task 16 -- K2 (hotpotqa). Tipsheets is NOT run: its loader"
echo "############ is not registered yet, which is part of Task 16 proper."
TASK=hotpotqa ./scripts/gate_kvcomm.sh

echo
echo "### ALL CHAINED GATES DONE at $(date -u +%FT%TZ)"
ls -la runs/
