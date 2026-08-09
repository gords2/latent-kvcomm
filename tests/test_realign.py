"""D4: realignment and renormalization are separate transforms.

Upstream ties both to one flag and applies the renorm unconditionally
(stock-latentmas/models.py:179-183 vs :204-213), so ablating the flag
ablates only the matmul. These tests pin the two apart.
"""
import torch

from lkv.realign import LatentProjector, build_realign_matrix


def test_matrix_solves_the_least_squares_system(tiny_qwen3):
    W, target_norm = build_realign_matrix(tiny_qwen3, "cpu")
    e_in = tiny_qwen3.get_input_embeddings().weight.detach().float()
    e_out = tiny_qwen3.get_output_embeddings().weight.detach().float()
    dim = e_in.shape[1]
    assert W.shape == (dim, dim)

    resid_w = (e_out @ W - e_in).norm()
    assert resid_w < (e_out - e_in).norm()      # beats the identity

    # Beating the identity is necessary but NOT sufficient, and this is the trap:
    # solving the TRANSPOSED problem also beats it (measured 2.97 vs 3.65 on this
    # fixture), so an orientation slip would pass an identity-only check. Compare
    # against the transposed solve directly.
    gram_t = e_in.T @ e_in + 1e-5 * torch.eye(dim)
    w_flipped = torch.linalg.solve(gram_t, e_in.T @ e_out)
    resid_flipped = (e_out @ w_flipped - e_in).norm()
    assert resid_w < resid_flipped, (resid_w.item(), resid_flipped.item())

    # target_norm is the mean row norm of the INPUT embedding
    assert torch.allclose(target_norm, e_in.norm(dim=1).mean(), atol=1e-4)


def test_realign_off_renorm_off_is_exactly_identity(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=False, device="cpu")
    h = torch.randn(3, tiny_qwen3.config.hidden_size)
    assert torch.equal(proj(h), h)
    # And the [D, D] solve must never have run for this configuration.
    assert proj.builds == 0


def test_renorm_alone_rescales_but_preserves_direction(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=False, renorm=True, device="cpu")
    h = torch.randn(3, tiny_qwen3.config.hidden_size)
    out = proj(h)
    _, target_norm = build_realign_matrix(tiny_qwen3, "cpu")
    assert torch.allclose(out.norm(dim=-1), target_norm.expand(3), atol=1e-3)
    cos = torch.nn.functional.cosine_similarity(out.float(), h.float(), dim=-1)
    assert torch.allclose(cos, torch.ones(3), atol=1e-5)


def test_realign_alone_changes_direction(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=True, renorm=False, device="cpu")
    h = torch.randn(3, tiny_qwen3.config.hidden_size)
    out = proj(h)
    cos = torch.nn.functional.cosine_similarity(out.float(), h.float(), dim=-1)
    assert (cos < 0.999).all(), "matmul had no effect -- matrix is the identity"


def test_shape_and_dtype_preserved(tiny_qwen3):
    for realign in (False, True):
        for renorm in (False, True):
            proj = LatentProjector(tiny_qwen3, realign=realign, renorm=renorm, device="cpu")
            h = torch.randn(2, tiny_qwen3.config.hidden_size, dtype=torch.float32)
            out = proj(h)
            assert out.shape == h.shape and out.dtype == h.dtype


def test_matrix_is_built_once(tiny_qwen3):
    proj = LatentProjector(tiny_qwen3, realign=True, renorm=True, device="cpu")
    h = torch.randn(1, tiny_qwen3.config.hidden_size)
    proj(h); proj(h)
    assert proj.builds == 1, "matrix rebuilt per call -- D x D solve is expensive"
