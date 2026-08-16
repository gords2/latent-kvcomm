#!/usr/bin/env bash
# Everything a fresh MI300X box needs, in one command. Run it FROM the repo
# root ON the box, after rsyncing the repo across.
#
# This exists because the recipe is not guessable and every step below cost
# real time to find:
#
#   * Python 3.11 is absent on this image and 3.12 ships WITHOUT ensurepip, so
#     `python3 -m venv` fails with a message that blames venv rather than the
#     missing package. Probing with `python3 -m venv --help` does not detect it
#     -- that succeeds regardless.
#   * rocm6.4 is the ONLY wheel index carrying a torch inside requirements.txt's
#     >=2.4,<2.9 pin. rocm7.x jumps to 2.10+, and plain PyPI serves the CUDA
#     wheel, which reports torch.cuda.is_available() == False on a working card.
#   * countries.jsonl and tipsheets.jsonl are not on the Hub and are not in this
#     repo. They only exist inside the KVComm source tree.
#   * The nltk corpora are load-bearing for F1: it lemmatises BOTH sides (D14).
#
# No Hugging Face account is needed anywhere. Qwen3 is Apache-2.0 and the KVComm
# gates use the ungated NousResearch Llama mirror.
set -euo pipefail

ROCM_INDEX=${ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.4}
KVCOMM_SRC=${KVCOMM_SRC:-/tmp/kvcomm}

echo "== GPU present? =="
# /dev/kfd is the definitive test. `nvidia-smi: command not found` on an AMD box
# proves nothing, and an `lspci` that was never installed reports a missing TOOL
# as a missing DEVICE -- that pair of non-facts once produced a confident,
# wrong "this droplet has no GPU" about a working MI300X.
ls -l /dev/kfd
rocm-smi --showproductname || echo "(rocm-smi absent -- /dev/kfd above is what counts)"

echo "== python3.12-venv =="
apt-get update -qq
apt-get install -y -qq python3.12-venv

echo "== venv + torch =="
PY=python3 TORCH_INDEX_URL="$ROCM_INDEX" ./scripts/setup_env.sh

echo "== nltk corpora =="
./.venv/bin/python -m nltk.downloader -q punkt punkt_tab wordnet omw-1.4 \
  averaged_perceptron_tagger averaged_perceptron_tagger_eng
# Assert rather than print: a partially-downloaded corpus set still imports and
# would then silently lemmatise to identity, quietly deflating every F1 score.
./.venv/bin/python - <<'EOF'
from lkv.lemma import lemmatize_text
got = lemmatize_text("cities were running faster mice")
assert got == "city be run faster mouse", f"lemmatiser wrong: {got!r}"
print("lemmatiser OK:", got)
EOF

echo "== bundled datasets =="
# Countries and Tipsheets ship inside the KVComm repo, not on the Hub.
[ -d "$KVCOMM_SRC" ] || git clone --depth 1 -q https://github.com/Zephyroam/KVComm.git "$KVCOMM_SRC"
mkdir -p lkv/data/files
cp "$KVCOMM_SRC/dataloader/data/countries.jsonl" \
   "$KVCOMM_SRC/dataloader/data/tipsheets.jsonl" lkv/data/files/
wc -l lkv/data/files/*.jsonl

echo "== suite =="
# NOT piped. A `| tail` here would report the pipeline's exit status, not
# pytest's, and a hard failure would come back as success.
./.venv/bin/python -m pytest -q

echo "== bootstrap OK =="
