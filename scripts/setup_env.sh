#!/usr/bin/env bash
# Bootstraps the environment. Works on the dev laptop (CPU) and the GPU box.
# The transformers pin is load-bearing -- see docs Global Constraints.
set -euo pipefail

PY=${PY:-python3.11}
command -v "$PY" >/dev/null || { echo "need $PY on PATH; set PY=..."; exit 1; }

"$PY" -m venv .venv
./.venv/bin/pip install -q --upgrade pip

# AMD (MI300X): `pip install torch` pulls the CUDA wheel from PyPI, which will
# not drive an AMD GPU -- torch.cuda.is_available() comes back False on a
# perfectly good card. Install the ROCm build FIRST, so the requirements pass
# below sees torch already satisfying its >=2.4,<2.9 pin and leaves it alone:
#
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.2 ./scripts/setup_env.sh
#
# Nothing else changes for AMD. torch's ROCm build maps the CUDA API onto HIP,
# so `--device cuda:0` and every torch.cuda.* call in lkv/instrument.py work
# unmodified. Leave this unset for NVIDIA or CPU.
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  echo "installing torch from $TORCH_INDEX_URL"
  ./.venv/bin/pip install -q --index-url "$TORCH_INDEX_URL" torch
fi

./.venv/bin/pip install -q -r requirements.txt
./.venv/bin/pip install -q -e .

./.venv/bin/python - <<'EOF'
import torch, transformers
assert transformers.__version__ == "4.53.3", transformers.__version__
print("transformers", transformers.__version__)
hip = getattr(torch.version, "hip", None)
print("torch", torch.__version__, "|",
      f"ROCm/HIP {hip}" if hip else f"CUDA {torch.version.cuda}")
print("accelerator available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1024**3:.0f} GiB")
else:
    # Loud: every gate would otherwise silently degrade to unusably slow CPU.
    print("WARNING: no accelerator visible. On AMD this usually means the CUDA "
          "wheel got installed instead of the ROCm one -- re-run with "
          "TORCH_INDEX_URL set (see the comment in this script).")
EOF
echo "env OK"
