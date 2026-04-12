import os
import sys
import unittest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main.models import ActorCritic, NUM_BINS, PopArtLayer

WINDOW_LEN = 16
TEMPORAL_DIM = 30
STATE_DIM = 38


def make_obs(B, device="cpu"):
    return {
        "temporal": torch.randn(B, WINDOW_LEN, TEMPORAL_DIM, device=device),
        "state": torch.randn(B, STATE_DIM, device=device),
        "masks": torch.zeros(B, 1, 2, 60, 80, device=device),
    }


class ModelInvariants(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.agent = ActorCritic()

    def test_actor_output_shape(self):
        obs = make_obs(8)
        logits = self.agent.actor(obs)
        self.assertEqual(tuple(logits.shape), (8, 4, NUM_BINS))

    def test_critic_output_shape(self):
        obs = make_obs(8)
        v = self.agent.get_value(obs)
        self.assertEqual(tuple(v.shape), (8, 1))

    def test_get_action_shapes(self):
        obs = make_obs(8)
        a, lp, idx = self.agent.get_action(obs, deterministic=False)
        self.assertEqual(tuple(a.shape), (8, 4))
        self.assertEqual(tuple(lp.shape), (8,))
        self.assertEqual(tuple(idx.shape), (8, 4))
        # action values must lie inside the categorical bin grid [-1, 1]
        self.assertTrue(torch.all(a >= -1.0 - 1e-6))
        self.assertTrue(torch.all(a <= 1.0 + 1e-6))

    def test_no_nan_after_100_forward_passes(self):
        # Domain truth: a randomly-initialized network must produce finite
        # outputs on bounded random inputs for at least 100 calls. NaN/Inf
        # within this budget indicates broken init or numerical instability.
        for _ in range(100):
            obs = make_obs(4)
            logits = self.agent.actor(obs)
            v = self.agent.get_value(obs)
            self.assertTrue(torch.isfinite(logits).all(), "NaN/Inf in actor logits")
            self.assertTrue(torch.isfinite(v).all(), "NaN/Inf in critic value")

    def test_gradient_flow_all_params(self):
        # Domain truth: every parameter must receive a non-None gradient
        # after a backward through both heads. A None grad means the param
        # is disconnected from the loss — a silent dead branch.
        obs = make_obs(4)
        self.agent.train()
        lp, ent, _, value, _, aux = self.agent.evaluate(obs, torch.zeros(4, 4, dtype=torch.long))
        loss = -lp.mean() - ent.mean() + value.pow(2).mean() + aux.pow(2).mean()
        loss.backward()
        missing = [n for n, p in self.agent.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [], f"params with no grad: {missing}")

    def test_popart_normalize_inverse(self):
        # Domain truth: popart.normalize_targets and popart.forward
        # are inverse transforms. Roundtrip must be identity.
        layer = PopArtLayer(8, 1)
        with torch.no_grad():
            layer.mu.fill_(3.7)
            layer.sigma.fill_(2.5)
        targets = torch.randn(64) * 5.0 + 1.0
        normed = layer.normalize_targets(targets)
        denormed = normed * layer.sigma + layer.mu
        self.assertTrue(torch.allclose(denormed.squeeze(), targets, atol=1e-5))

    def test_popart_update_preserves_value(self):
        # Domain truth: PopArt's weight correction is designed so that
        # the network output `forward(x)` is INVARIANT under update_stats.
        # If a stats update changes the unnormalized prediction, the
        # correction is broken and the critic will see discontinuities.
        layer = PopArtLayer(8, 1)
        x = torch.randn(16, 8)
        before = layer(x).detach().clone()
        targets = torch.randn(256) * 10.0 + 5.0
        layer.update_stats(targets)
        after = layer(x).detach()
        self.assertTrue(torch.allclose(before, after, atol=1e-4),
                        f"max diff {(before - after).abs().max().item():.6f}")


if __name__ == "__main__":
    unittest.main()
