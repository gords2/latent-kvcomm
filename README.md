# latent-kvcomm

KVComm layer-wise KV trimming applied to LatentMAS inter-agent cache handoff.

Clean-room reimplementation of two published systems so each reproduces its own
paper's accuracy, after which KVComm's layer-selective KV-cache trimming is
applied at LatentMAS's inter-agent boundaries.

- **[docs/NEXT.md](docs/NEXT.md)** — **start here.** Current state, what runs next, and the rulings that bind.
- **[docs/FINDINGS.md](docs/FINDINGS.md)** — measured defects and every deviation from upstream. Read before citing any number.
- **[the plan](docs/superpowers/plans/2026-08-09-latentmas-kvcomm.md)** — the 21-task spec.

## Status

All CPU-testable work is complete and reviewed: 142 tests, no GPU or network
required. The GPU accuracy gates and the integration sweep are deferred — see
`docs/NEXT.md`.

```bash
PY=python3.11 ./scripts/setup_env.sh
./.venv/bin/python -m pytest -q
```

`transformers==4.53.3` is a hard pin, not a floor.
