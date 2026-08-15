#!/usr/bin/env bash
# Gate L1 -- GSM8K on Qwen3-8B, sweeping latent_steps.
#
# Reproduction uses the paper's SAMPLED decoding (temperature 0.6, top_p 0.95)
# because that is what the published numbers were produced with. Internal A/B
# comparisons elsewhere use --greedy; never mix the two inside one comparison.
#
# latent_steps is recorded NOWHERE upstream: its argparse default is 0, which is
# degenerate (the relay cache is discarded), and the two shipped example logs
# show 0 and 10, neither matching the table cell it accompanies. The paper says
# tune over [0, 80], so it is searched here rather than assumed.
#
# No -e: one failing condition must not abandon the rest of the sweep. Each run
# is skipped if its JSON already exists, so this is safe to re-run after an
# interruption without repeating hours of work.
set -uo pipefail
PY=./.venv/bin/python
OUT=runs; mkdir -p "$OUT"
M=${M:-Qwen/Qwen3-8B}
N=${N:-200}
BS=${BS:-16}

run() {  # run <tag> <extra args...>
  local tag=$1; shift
  local f="$OUT/${tag}.json"
  [ -f "$f" ] && { echo "skip $tag (exists)"; return 0; }
  echo "=== $tag ==="
  $PY -m lkv.run --task gsm8k --model_name "$M" --max_samples "$N" \
      --generate_bs "$BS" --seed 42 --out_json "$f" "$@" 2>&1 | tail -20
}

for S in 0 10 20 40 80; do
  run "lm_gsm8k_8b_s${S}" --system latentmas --latent_steps "$S"
done
run lm_gsm8k_8b_single --system single --latent_steps 0

$PY - <<'EOF'
import glob, json

rows = []
for path in sorted(glob.glob("runs/lm_gsm8k_8b_*.json")):
    d = json.load(open(path))
    rows.append({"run": path.split("/")[-1], "system": d["system"],
                 "steps": d["latent_steps"], "acc": d["accuracy"],
                 "n": d["n"], "sec": d["sec_per_item"]})

print(f"{'run':30} {'system':10} {'steps':>5} {'n':>4} {'acc':>7} {'s/item':>7}")
for r in rows:
    print(f"{r['run']:30} {r['system']:10} {r['steps']:5} {r['n']:4} "
          f"{r['acc']:7.4f} {r['sec']:7.2f}")

relay = [r for r in rows if r["system"] == "latentmas"]
single = [r for r in rows if r["system"] == "single"]

print("\nTARGET  latentmas 0.938 | single 0.811")
if not relay or not single:
    print("INCOMPLETE: need both a latentmas sweep and a single-agent run.")
    raise SystemExit(0)

# max() over dicts, keyed on accuracy. The plan's `max(r for r in rows ...)`
# compared TUPLES, so it picked the lexicographically largest FILENAME rather
# than the best accuracy -- and then never used the result at all.
best = max(relay, key=lambda r: r["acc"])
s = single[0]
print(f"BEST    latent_steps={best['steps']} acc={best['acc']:.4f} ({best['run']})")
print(f"SINGLE  acc={s['acc']:.4f}")

# L1: some latent_steps >= 0.91 at n=200 (0.938 - 1.5*SE, SE ~ 0.018), AND
# single-agent within [0.76, 0.86].
ok_relay = best["acc"] >= 0.91
ok_single = 0.76 <= s["acc"] <= 0.86
print(f"\nL1 relay  >= 0.910       : {'PASS' if ok_relay else 'FAIL'} ({best['acc']:.4f})")
print(f"L1 single in [0.76, 0.86]: {'PASS' if ok_single else 'FAIL'} ({s['acc']:.4f})")
print("GATE L1: " + ("PASS" if (ok_relay and ok_single) else "FAIL"))
if not (ok_relay and ok_single):
    print("\nDiagnostic ladder is in the plan's Task 10 Step 3. Work it IN ORDER "
          "and record each outcome in docs/FINDINGS.md before trying the next.")
EOF
