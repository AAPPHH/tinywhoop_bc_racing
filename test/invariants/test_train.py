import os
import sys
import tempfile
import unittest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main.models import ActorCritic, PopArtLayer
from main.train import compute_vtrace


class TrainInvariants(unittest.TestCase):

    def test_vtrace_finite_on_random_input(self):
        # Domain truth: V-Trace targets and advantages must be finite for
        # any bounded inputs. Inf/NaN here means a downstream loss explosion.
        torch.manual_seed(0)
        B, T = 4, 16
        behavior_lp = torch.randn(B, T) * 0.5
        current_lp = torch.randn(B, T) * 0.5
        rewards = torch.randn(B, T) * 0.1
        values = torch.randn(B, T)
        dones = torch.zeros(B, T)
        dones[:, -1] = 1.0
        bootstrap = torch.randn(B)
        targets, advs, rho = compute_vtrace(
            behavior_lp, current_lp, rewards, values, dones, bootstrap,
            gamma=0.99, rho_bar=1.0, c_bar=1.0,
        )
        self.assertTrue(torch.isfinite(targets).all(), "vtrace targets contain NaN/Inf")
        self.assertTrue(torch.isfinite(advs).all(), "vtrace advantages contain NaN/Inf")
        self.assertTrue(torch.isfinite(torch.tensor(rho)), "rho is NaN/Inf")

    def test_vtrace_zero_reward_zero_advantage(self):
        # Domain truth: with rewards=0, gamma=1, dones=0, values=const, and
        # on-policy (current_lp == behavior_lp → rho=c=1), V-Trace targets
        # collapse to V(s) and advantages collapse to 0.
        B, T = 2, 8
        lp = torch.zeros(B, T)
        rewards = torch.zeros(B, T)
        values = torch.full((B, T), 3.5)
        dones = torch.zeros(B, T)
        bootstrap = torch.full((B,), 3.5)
        targets, advs, _ = compute_vtrace(
            lp, lp, rewards, values, dones, bootstrap,
            gamma=1.0, rho_bar=1.0, c_bar=1.0,
        )
        self.assertTrue(torch.allclose(targets, torch.full_like(targets, 3.5), atol=1e-5))
        self.assertTrue(torch.allclose(advs, torch.zeros_like(advs), atol=1e-5))

    def test_popart_sigma_positive_after_1000_updates(self):
        # Domain truth: PopArt sigma must remain strictly positive (clamped at 1e-4)
        # under any sequence of return distributions. Sigma=0 would create
        # division-by-zero in normalize_targets.
        torch.manual_seed(1)
        layer = PopArtLayer(16, 1)
        for _ in range(1000):
            r = torch.randn(64) * (0.01 + torch.rand(1).item() * 50.0)
            layer.update_stats(r)
            self.assertTrue(torch.isfinite(layer.mu).all(), "popart mu went non-finite")
            self.assertTrue(torch.isfinite(layer.sigma).all(), "popart sigma went non-finite")
            self.assertGreater(layer.sigma.item(), 0.0, "popart sigma collapsed to 0")

    def test_checkpoint_roundtrip(self):
        # Domain truth: torch.save → torch.load → load_state_dict must
        # restore parameters bit-for-bit. A mismatch means resume is broken
        # and any "continuing training" silently restarts from random.
        torch.manual_seed(2)
        agent = ActorCritic()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            torch.save({"model_state_dict": agent.state_dict()}, path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            agent2 = ActorCritic()
            agent2.load_state_dict(ckpt["model_state_dict"])
            for (n1, p1), (n2, p2) in zip(agent.named_parameters(), agent2.named_parameters()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.equal(p1, p2), f"param mismatch on {n1}")
            for (n1, b1), (n2, b2) in zip(agent.named_buffers(), agent2.named_buffers()):
                self.assertEqual(n1, n2)
                self.assertTrue(torch.equal(b1, b2), f"buffer mismatch on {n1}")


if __name__ == "__main__":
    unittest.main()
