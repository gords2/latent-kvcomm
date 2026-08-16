#!/usr/bin/env bash
# Phase 5: KVComm trimming applied at LatentMAS's inter-agent boundaries.
#
# --greedy throughout: these are internal A/B comparisons, and mixing sampled
# and greedy runs inside one comparison invalidates the pairing.
#
# TWO THINGS A READER MUST KNOW BEFORE USING THESE NUMBERS.
#
# 1. The premise of this sweep is not satisfied. The plan picks the task with
#    the larger relay-versus-single gap, "the only one where trimming has
#    something to damage". Measured, single-agent beats the relay on BOTH
#    tasks (GSM8K 0.960 vs 0.950; MBPP+ 0.7566 vs 0.7487), so there is no
#    relay advantage for trimming to erode. What this sweep still measures
#    honestly is accuracy against cache bytes and attention positions. What it
#    CANNOT measure is preservation of a benefit that was never observed.
#
# 2. CAUSAL defaults to 1. Measured on Countries, the stock non-causal path
#    scores 0.150 where the corrected path scores 0.615 at the same retention
#    (D8, docs/FINDINGS.md). Every integration number should be produced on
#    the corrected path; CAUSAL=0 exists to measure the gap, not to reproduce.
set -uo pipefail
PY=./.venv/bin/python
OUT=runs; mkdir -p "$OUT"
M=${M:-Qwen/Qwen3-8B}
TASK=${TASK:-mbppplus}
N=${N:-378}
BS=${BS:-8}
# latent_steps. The sweep over [0,80] was flat within 1-2 SE (ruling 2), so
# this is not a tuned optimum -- it is pinned to whatever the existing relay
# reference for this task used, so the sweep stays paired against it.
S_BEST=${S_BEST:-0}
CAUSAL=${CAUSAL:-1}
CAUSAL_FLAG=""
[ "$CAUSAL" = "1" ] && CAUSAL_FLAG="--causal_ragged"

run() {
  local tag=$1; shift
  local f="$OUT/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag (exists)"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --task "$TASK" --model_name "$M" --max_samples "$N" \
      --latent_steps "$S_BEST" --greedy --seed 42 --generate_bs "$BS" \
      --out_json "$f" "$@" 2>&1 | tail -15
}

# Reference points: untrimmed relay, and one agent.
run "int_${TASK}_full"   --system latentmas --top_layers 1.0
run "int_${TASK}_single" --system single

# Two scopes x three ratios, trimming at EVERY boundary so the loss compounds.
for SCOPE in all delta; do
  for R in 0.3 0.5 0.7; do
    run "int_${TASK}_${SCOPE}_r${R/0./}0" --system latentmas --top_layers "$R" \
        --trim_scope "$SCOPE" --trim_every_hop --calib_source relay $CAUSAL_FLAG
  done
done

# One-shot (first boundary only), to separate compounding from trimming itself.
run "int_${TASK}_all_r50_oneshot" --system latentmas --top_layers 0.5 \
    --trim_scope all --calib_source relay $CAUSAL_FLAG

# Control: random layer choice at the same retention. If random matches the
# ranked selection, the ranking is doing no work.
run "int_${TASK}_all_r50_random" --system latentmas --top_layers 0.5 \
    --trim_scope all --trim_every_hop --random_selection $CAUSAL_FLAG

# Control: pair-style calibration imported from the Task 14 distribution.
# NOT RUNNABLE on gsm8k/mbppplus -- kvcomm_sender_prompt(instruction, context)
# needs a `context` field those items do not have, and config.py rejects the
# combination at parse time rather than letting it KeyError mid-run. Announced
# rather than silently skipped: a dropped condition that prints nothing reads
# as "swept everything" when the table is written up later.
case "$TASK" in
  gsm8k|mbppplus)
    echo "SKIP int_${TASK}_all_r50_paircal: --calib_source pair requires a"
    echo "  task whose items carry 'context'. ${TASK} items carry only"
    echo "  question/gold. Run this control on countries/hotpotqa/tipsheets."
    ;;
  *)
    run "int_${TASK}_all_r50_paircal" --system latentmas --top_layers 0.5 \
        --trim_scope all --trim_every_hop --calib_source pair $CAUSAL_FLAG
    ;;
esac

$PY scripts/analyze.py "$TASK"
