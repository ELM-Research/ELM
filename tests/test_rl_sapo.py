import torch
from types import SimpleNamespace

from rl.losses import compute_rl_loss


def test_compute_rl_loss_sapo_runs():
    torch.manual_seed(0)
    bsz, seq, vocab = 2, 6, 11
    labels = torch.tensor([[1, 2, 3, -100, -100, -100], [1, 4, 5, 6, -100, -100]])
    out = SimpleNamespace(logits=torch.randn(bsz, seq, vocab))
    target_shape = (bsz, seq - 1)
    batch = {
        "elm_labels": labels,
        "old_log_prob": torch.zeros(target_shape),
        "advantages": torch.ones(target_shape),
    }
    args = SimpleNamespace(rl_loss="sapo", rl_loss_agg_mode="seq-mean-token-mean", sapo_tau_pos=1.0, sapo_tau_neg=1.1, batch_size=bsz, world_size=1)
    loss, metrics = compute_rl_loss(batch, out, args)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "actor/ppo_kl" in metrics
