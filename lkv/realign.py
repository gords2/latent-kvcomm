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
        if self._w is None and (self.realign or self.renorm):
            self._w, self._target_norm = build_realign_matrix(self._model, self.device)
            self.builds += 1

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
