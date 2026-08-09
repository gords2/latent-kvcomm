#!/usr/bin/env bash
# Bootstraps the environment. Works on the dev laptop (CPU) and the GPU box.
# The transformers pin is load-bearing -- see docs Global Constraints.
set -euo pipefail

PY=${PY:-python3.11}
command -v "$PY" >/dev/null || { echo "need $PY on PATH; set PY=..."; exit 1; }

"$PY" -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/pip install -q -e .

./.venv/bin/python - <<'EOF'
import torch, transformers
assert transformers.__version__ == "4.53.3", transformers.__version__
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
EOF
echo "env OK"
