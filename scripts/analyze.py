"""Table for the Phase 5 integration sweep: accuracy against what was saved.

Usage: python scripts/analyze.py [task]

The point of the integration is a trade: trimming the relay cache buys memory
and costs accuracy. A table of accuracies alone cannot show a trade, so every
row also carries the measured cache bytes and attention positions at the final
relay hop, expressed as a fraction of the untrimmed reference run.
"""
import glob
import json
import os
import sys

task = sys.argv[1] if len(sys.argv) > 1 else "mbppplus"

rows = {}
for path in sorted(glob.glob(f"runs/int_{task}_*.json")):
    rows[os.path.basename(path)[: -len(".json")]] = json.load(open(path))

if not rows:
    print(f"no runs/int_{task}_*.json found")
    raise SystemExit(0)

reference = rows.get(f"int_{task}_full")
ref_bytes = (reference or {}).get("cache_bytes_at_final_hop_mean") or 0.0
ref_pos = (reference or {}).get("attention_positions_at_final_hop_mean") or 0.0


def frac(value, ref):
    """Fraction of the untrimmed reference, or None when it cannot be formed.

    Returns None rather than 1.0 or 0.0 when the reference is missing or zero:
    a made-up ratio in this column would read as a measured saving.
    """
    if not ref or value is None:
        return None
    return value / ref


def cell(x, width=9):
    return " " * (width - 3) + "n/a" if x is None else f"{x:{width}.4f}"


print(f"\n{'run':38} {'acc':>8} {'layer_ret':>10} {'bytes/ref':>10} "
      f"{'pos/ref':>9} {'batches':>8}  path")
for name in sorted(rows):
    d = rows[name]
    # NOT d.get("layer_retention", 1.0): systems with no relay carry the key
    # with an explicit None, and .get's default only covers a MISSING key --
    # the default form returns None and then formatting it raises (ruling 1).
    ret = d.get("layer_retention")
    print(f"{name:38} {d['accuracy']:8.4f} {cell(ret, 10)} "
          f"{cell(frac(d.get('cache_bytes_at_final_hop_mean'), ref_bytes), 10)} "
          f"{cell(frac(d.get('attention_positions_at_final_hop_mean'), ref_pos), 9)} "
          f"{d.get('cache_stat_batches', 0):8}  {d.get('ragged_path', '?')}")

full = rows.get(f"int_{task}_full")
single = rows.get(f"int_{task}_single")
if full and single:
    print(f"\nrelay(untrimmed)={full['accuracy']:.4f}  "
          f"single={single['accuracy']:.4f}  "
          f"delta={full['accuracy'] - single['accuracy']:+.4f}")
    if full["accuracy"] <= single["accuracy"]:
        print("  NOTE: the relay does not beat one agent here, so this sweep")
        print("  measures the cost of trimming, NOT the preservation of a")
        print("  multi-agent benefit. Do not report it as the latter.")

oneshot = rows.get(f"int_{task}_all_r50_oneshot")
every = rows.get(f"int_{task}_all_r50")
if oneshot and every:
    print(f"\ncompounding: one-shot={oneshot['accuracy']:.4f} vs "
          f"every-hop={every['accuracy']:.4f} at the same 0.5 retention "
          f"(delta={oneshot['accuracy'] - every['accuracy']:+.4f}). A gap here "
          "is the cost of trimming REPEATEDLY, separate from trimming at all.")

random_run = rows.get(f"int_{task}_all_r50_random")
if random_run and every:
    print(f"\nranking vs random at 0.5: ranked={every['accuracy']:.4f} "
          f"random={random_run['accuracy']:.4f} "
          f"(delta={every['accuracy'] - random_run['accuracy']:+.4f}). If random "
          "matches the ranked selection, the importance ranking is doing no work "
          "in the relay setting even though it does in the pair setting.")

missing = [t for t in (f"int_{task}_full", f"int_{task}_single") if t not in rows]
if missing:
    print(f"\nINCOMPLETE -- missing reference runs: {missing}")
