#!/usr/bin/env bash
# The Task 20 sweep, with its conditions run CONCURRENTLY instead of one at a
# time. Same conditions, same arguments, same output files as
# scripts/sweep_integration.sh -- only the scheduling differs.
#
# Why this exists: MBPP+ is 378 items at max_new_tokens=4096, roughly 100-140
# minutes per condition. Ten conditions in sequence is 17-23 hours of wall
# clock on a card that sits at ~57% VRAM while running one of them.
# Autoregressive decode is memory-bandwidth-bound, so a single stream cannot
# saturate an MI300X; concurrency is the only lever that helps here.
#
# BATCH SIZE IS NOT THAT LEVER. Raising --generate_bs changes the order in
# which items consume the RNG stream, so a sampled run stops being comparable
# with everything already measured. These runs are --greedy, where that does
# not apply, but the batch size is left alone anyway so the sweep stays
# comparable with the greedy references it is paired against.
#
# JOBS is capped by VRAM, not by cores: each 8B stream holds roughly 16-38 GB
# of the 192 GB card, and an MBPP+ cache grows throughout a 4096-token
# generation. Leave headroom -- an OOM two hours into a condition costs more
# than the concurrency gains.
set -uo pipefail
PY=./.venv/bin/python
OUT=runs; mkdir -p "$OUT"
M=${M:-Qwen/Qwen3-8B}
TASK=${TASK:-mbppplus}
N=${N:-378}
BS=${BS:-8}
S_BEST=${S_BEST:-0}
JOBS=${JOBS:-4}
CAUSAL=${CAUSAL:-1}
CAUSAL_FLAG=""
[ "$CAUSAL" = "1" ] && CAUSAL_FLAG="--causal_ragged"

# THINK=0 passes --no_thinking to every condition INCLUDING the two reference
# runs, so the sweep stays internally paired.
#
# This is the configuration the sweep should actually run in, and the reason is
# measured rather than assumed. The plan says to sweep whichever task shows the
# larger paired relay-versus-single gap -- "the only one where trimming has
# something to damage". Of the four cells measured, exactly one has a real gap:
#
#   task    thinking  relay   single  delta    McNemar p
#   GSM8K   on        0.9500  0.9600  -0.0100  0.6875   ns
#   GSM8K   off       0.9150  0.8900  +0.0250  0.3018   ns
#   MBPP+   on        0.7487  0.7566  -0.0079  0.7201   ns
#   MBPP+   off       0.7302  0.6931  +0.0370  0.0201   SIGNIFICANT
#
# So MBPP+ with THINK=0 is the only setting where a relay advantage exists for
# trimming to erode. Sweeping GSM8K would measure how much trimming degrades a
# task one agent already solves at 0.96 -- a real number answering no question.
# It is also ~4x cheaper than MBPP+ with thinking left on.
THINK=${THINK:-1}
THINK_FLAG=""
[ "$THINK" = "0" ] && THINK_FLAG="--no_thinking"

mkdir -p runs/logs

# One condition per line: "<tag>|<extra args>". Kept as data rather than a
# nest of loops so the concurrency layer below stays trivial to read.
CONDITIONS=$(
  echo "int_${TASK}_full|--system latentmas --top_layers 1.0 $THINK_FLAG"
  echo "int_${TASK}_single|--system single $THINK_FLAG"
  for SCOPE in all delta; do
    for R in 0.3 0.5 0.7; do
      echo "int_${TASK}_${SCOPE}_r${R/0./}0|--system latentmas --top_layers $R --trim_scope $SCOPE --trim_every_hop --calib_source relay $CAUSAL_FLAG $THINK_FLAG"
    done
  done
  echo "int_${TASK}_all_r50_oneshot|--system latentmas --top_layers 0.5 --trim_scope all --calib_source relay $CAUSAL_FLAG $THINK_FLAG"
  echo "int_${TASK}_all_r50_random|--system latentmas --top_layers 0.5 --trim_scope all --trim_every_hop --random_selection $CAUSAL_FLAG $THINK_FLAG"
  case "$TASK" in
    gsm8k|mbppplus) : ;;   # announced below, not silently dropped
    *) echo "int_${TASK}_all_r50_paircal|--system latentmas --top_layers 0.5 --trim_scope all --trim_every_hop --calib_source pair $CAUSAL_FLAG $THINK_FLAG" ;;
  esac
)

case "$TASK" in
  gsm8k|mbppplus)
    echo "SKIP int_${TASK}_all_r50_paircal: --calib_source pair needs items"
    echo "  carrying 'context'; ${TASK} items carry only question/gold, and"
    echo "  config.py rejects the pair at parse time. Run that control on"
    echo "  countries/hotpotqa/tipsheets."
    ;;
esac

run_one() {
  local line=$1
  local tag=${line%%|*}
  local args=${line#*|}
  local f="runs/${tag}.json"
  if [ -f "$f" ]; then echo "skip $tag (exists)"; return 0; fi
  echo "start $tag"
  # Each condition gets its own log. A shared stdout would interleave several
  # concurrent tqdm streams into noise.
  # shellcheck disable=SC2086
  $PY -m lkv.run --task "$TASK" --model_name "$M" --max_samples "$N" \
      --latent_steps "$S_BEST" --greedy --seed 42 --generate_bs "$BS" \
      --out_json "$f" $args > "runs/logs/${tag}.log" 2>&1
  local status=$?
  # Captured explicitly. Piping this into tail would report the PIPELINE's
  # status and turn a hard failure into a silent success -- that has already
  # happened once in this project.
  if [ $status -eq 0 ]; then
    echo "ok   $tag"
  else
    echo "FAIL $tag (exit $status, see runs/logs/${tag}.log)"
  fi
  return 0
}
export -f run_one
export PY OUT M TASK N BS S_BEST

echo "$CONDITIONS" | grep . | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo "=== all conditions finished ==="
$PY scripts/analyze.py "$TASK"
