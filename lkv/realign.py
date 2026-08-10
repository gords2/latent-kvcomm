"""LatentMAS latent-space realignment, with the renorm split out.

The matrix is the ridge-regularized least-squares solution of E_out W ~= E_in:
    W = (E_out^T E_out + 1e-5 I)^-1 (E_out^T E_in)
solved in fp32, exactly as stock-latentmas/models.py:158-185. target_norm is the
mean L2 row norm of the INPUT embedding (models.py:177).

Upstream applies the renorm on every latent step regardless of the
--latent_space_realign flag; here `realign` gates the matmul and `renorm` gates
the rescale, independently, so either can be ablated without the other.
"""
from __future__ import annotations

import torch


def compute_target_norm(model, device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The mean L2 row norm of the INPUT embedding alone (models.py:177).

    Split out of build_realign_matrix so the renorm-only path (renorm=True,
    realign=False) never has to pay for the [D, D] gram + solve: target_norm
    does not depend on the output embedding, W, or the solve at all, so
    computing it standalone is both cheaper and simpler to audit than
    discarding half of build_realign_matrix's return value.
    """
    input_embeds = model.get_input_embeddings()
    if input_embeds is None:
        raise RuntimeError("cannot compute target_norm: input embedding not accessible")
    e_in = input_embeds.weight.detach().to(device=device, dtype=dtype)
    return e_in.norm(dim=1).mean().detach()


def build_realign_matrix(model, device, dtype: torch.dtype = torch.float32):
    input_embeds = model.get_input_embeddings()
    output_embeds = model.get_output_embeddings() or getattr(model, "lm_head", None)
    if input_embeds is None or output_embeds is None:
        raise RuntimeError("cannot build realign matrix: embedding weights not accessible")

    e_in = input_embeds.weight.detach().to(device=device, dtype=dtype)
    e_out = output_embeds.weight.detach().to(device=device, dtype=dtype)

    gram = e_out.T @ e_out
    gram = gram + 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    rhs = e_out.T @ e_in
    w = torch.linalg.solve(gram, rhs)
    target_norm = e_in.norm(dim=1).mean().detach()
    return w, target_norm


class LatentProjector:
    """Applies (optionally) the realign matmul and (optionally) the renorm."""

    def __init__(self, model, realign: bool, renorm: bool, device):
        self._model = model
        self.realign = realign
        self.renorm = renorm
        self.device = device
        self._w = None
        self._target_norm = None
        self.builds = 0

    def _ensure(self):
        if self._target_norm is not None or not (self.realign or self.renorm):
            return
        if self.realign:
            # Needs W, so pay for the full [D, D] gram + solve -- there is no
            # cheaper way to get a matmul.
            self._w, self._target_norm = build_realign_matrix(self._model, self.device)
            self.builds += 1
        else:
            # renorm alone never reads self._w (see __call__), so skip the
            # gram + solve entirely: at Qwen3-8B that solve plus the two fp32
            # embedding copies it needs are a multi-GB, multi-second transient
            # this configuration has no use for.
            self._target_norm = compute_target_norm(self._model, self.device)

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.realign and not self.renorm:
            return hidden
        self._ensure()
        out_dtype = hidden.dtype
        x = hidden.to(torch.float32)
        if self.realign:
            x = x @ self._w.to(x.device)
        if self.renorm:
            scale = self._target_norm.to(x.device) / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x * scale
        return x.to(out_dtype)
