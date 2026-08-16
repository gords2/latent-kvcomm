#!/usr/bin/env bash
# Gates K1 (countries) and K2 (hotpotqa / tipsheets).
#
# KVComm decodes GREEDILY (stock-kvcomm/eval.py:80-88 sets do_sample=False), so
# --greedy here is reproduction, not an internal control. That is the opposite
# of gate_latentmas.sh, which samples because LatentMAS's published numbers were
# produced that way. Never mix the two inside one comparison.
#
# M defaults to the UNGATED mirror: meta-llama/* needs a Hugging Face account,
# and KVComm's published numbers are Llama numbers, so the mirror is what keeps
# this gate reachable at all. See docs/FINDINGS.md's mirror deviation note.
set -uo pipefail
PY=./.venv/bin/python
OUT=runs; mkdir -p "$OUT"
M=${M:-NousResearch/Meta-Llama-3.1-8B-Instruct}
TASK=${TASK:-countries}

# Whether the trimmed conditions decode with per-layer causal masks (D8).
#
# Defaults ON, reversing this project's earlier assumption. The reasoning used
# to be "off reproduces KVComm as published, since their table came from the
# stock path" -- falsified by measurement (Countries, n=200, greedy, the
# Llama-3.1-8B mirror):
#
#   ratio | stock  | causal | published
#   0.3   | 0.030  | 0.315  | 0.51
#   0.5   | 0.150  | 0.615  | 0.62
#   0.7   | 0.155  | 0.620  | 0.62
#
# The non-causal path reproduces nothing; the corrected one lands on two of the
# three published cells. Set CAUSAL=0 to measure the stock path -- that is the
# D8 gap experiment, not the reproduction.
CAUSAL=${CAUSAL:-1}
CAUSAL_FLAG=""
[ "$CAUSAL" = "1" ] && CAUSAL_FLAG="--causal_ragged"

run() {
  local tag=$1; shift
  local f="$OUT/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag (exists)"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --task "$TASK" --model_name "$M" --greedy --seed 42 \
      --calib_size 1 --mu 0.5 --sigma 10.0 --out_json "$f" "$@" 2>&1 | tail -20
}

run "kv_${TASK}_baseline" --system baseline
run "kv_${TASK}_skyline"  --system skyline
for R in 0.3 0.5 0.7; do
  run "kv_${TASK}_r${R/0./}0" --system kvcomm --top_layers "$R" $CAUSAL_FLAG
done
run "kv_${TASK}_rand50" --system kvcomm --top_layers 0.5 --random_selection $CAUSAL_FLAG

TASK="$TASK" $PY - <<'EOF'
import glob, json, os

task = os.environ["TASK"]
print(f"\n{'run':32} {'acc(F1>0.5)':>12} {'meanF1':>8} {'layer_ret':>10} {'path':>16}")
rows = {}
for path in sorted(glob.glob(f"runs/kv_{task}_*.json")):
    d = json.load(open(path))
    # NOT d.get("layer_retention", 1.0): baseline and skyline records carry that
    # key with an explicit None (neither has a relay or trim concept), and .get's
    # default applies only when the key is ABSENT. The default form returns None
    # and then f"{None:10.4f}" raises TypeError -- the gotcha recorded as ruling
    # 1 in docs/NEXT.md, which the plan's own snippet still contains.
    ret = d.get("layer_retention")
    ret_s = "n/a" if ret is None else f"{ret:.4f}"
    name = os.path.basename(path)
    rows[name] = d
    print(f"{name:32} {d['accuracy']:12.4f} {d['mean_score']:8.4f} "
          f"{ret_s:>10} {d.get('ragged_path', '?'):>16}")


def get(sfx, field):
    d = rows.get(f"kv_{task}_{sfx}.json")
    return None if d is None else d[field]


# Published Countries column. The D10 decision: whichever of the two metric
# columns tracks this row across all five conditions is what KVComm reported.
PUBLISHED = {
    "countries": {"baseline": 0.00, "skyline": 0.62,
                  "r30": 0.51, "r50": 0.62, "r70": 0.62},
}
pub = PUBLISHED.get(task)
if pub:
    print("\nPUBLISHED " + task + ": "
          + "  ".join(f"{k}={v:.2f}" for k, v in pub.items()))

    for field, label in (("accuracy", "acc(F1>0.5)"), ("mean_score", "meanF1")):
        vals = {k: get(k, field) for k in pub}
        if any(v is None for v in vals.values()):
            print(f"{label:12} INCOMPLETE")
            continue
        dev = max(abs(vals[k] - pub[k]) for k in pub)
        print(f"{label:12} " + "  ".join(f"{k}={vals[k]:.2f}" for k in pub)
              + f"   max|dev|={dev:.3f}")

    # K1: baseline <= 0.05, skyline in [0.55, 0.69], r30 >= 0.42, r50 >= 0.55,
    # monotone r30 <= r50 <= r70 with 0.02 tolerance for the 0.5/0.7 tie the
    # paper itself reports.
    for field, label in (("accuracy", "acc(F1>0.5)"), ("mean_score", "meanF1")):
        v = {k: get(k, field) for k in pub}
        if any(x is None for x in v.values()):
            continue
        checks = [
            ("baseline <= 0.05", v["baseline"] <= 0.05),
            ("skyline in [0.55, 0.69]", 0.55 <= v["skyline"] <= 0.69),
            ("r30 >= 0.42", v["r30"] >= 0.42),
            ("r50 >= 0.55", v["r50"] >= 0.55),
            ("monotone r30 <= r50 <= r70 (0.02 tol)",
             v["r30"] <= v["r50"] + 0.02 and v["r50"] <= v["r70"] + 0.02),
        ]
        print(f"\nGATE K1 under {label}:")
        for name, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"  => {'PASS' if all(o for _, o in checks) else 'FAIL'}")

# K2 (hotpotqa/tipsheets) has no PUBLISHED row here -- the paper's per-task
# column for it was never transcribed into this repo, and inventing one to
# print a max|dev| against would manufacture a comparison. The thresholds
# below are the ones fixed in docs/NEXT.md before any of it was measured,
# which is what keeps them a gate rather than a description of the result.
K2_THRESHOLDS = {
    "hotpotqa": [
        ("baseline in [0.13, 0.25]", lambda v: 0.13 <= v["baseline"] <= 0.25),
        ("skyline >= 0.66", lambda v: v["skyline"] >= 0.66),
        ("r30 >= 0.26", lambda v: v["r30"] >= 0.26),
        ("r50 >= 0.53", lambda v: v["r50"] >= 0.53),
        ("r70 >= 0.62", lambda v: v["r70"] >= 0.62),
        ("strictly increasing r30 < r50 < r70",
         lambda v: v["r30"] < v["r50"] < v["r70"]),
    ],
}
if task in K2_THRESHOLDS:
    for field, label in (("accuracy", "acc(F1>0.5)"), ("mean_score", "meanF1")):
        v = {k: get(k, field) for k in ("baseline", "skyline", "r30", "r50", "r70")}
        if any(x is None for x in v.values()):
            print(f"\nGATE K2 under {label}: INCOMPLETE")
            continue
        print(f"\nGATE K2 under {label}:")
        results = [(name, bool(fn(v))) for name, fn in K2_THRESHOLDS[task]]
        for name, ok in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"  => {'PASS' if all(o for _, o in results) else 'FAIL'}")

rand, r50 = get("rand50", "accuracy"), get("r50", "accuracy")
if rand is not None and r50 is not None:
    print(f"\nrand50={rand:.4f} vs r50={r50:.4f}. If random matches or beats "
          "KVComm the ranking is not working -- but the paper itself reports "
          "random winning at ratio 0.7 on 5 of 8 datasets, so a tie there is "
          "expected and is NOT evidence of a bug.")
EOF
