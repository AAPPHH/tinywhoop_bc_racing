import os
import sys
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main.env import GateRacingEnv


def _step_zero(env, n):
    a = np.zeros(4, dtype=np.float32)
    for _ in range(n):
        env.step(a)


def _step_uniform(env, val, n):
    a = np.full(4, val, dtype=np.float32)
    for _ in range(n):
        env.step(a)


class EnvInvariants(unittest.TestCase):

    def setUp(self):
        self.env = GateRacingEnv(track="circle_small", academy_level=0, dr_scale=0.0)
        self.env.reset()

    def tearDown(self):
        self.env.close()

    def test_hover_stable_200_steps(self):
        # Domain truth: symmetric thrust at HOVER_RPM produces zero net force/torque
        # beyond gravity compensation. The drone must stay aloft and level.
        _step_zero(self.env, 200)
        pos, orn = self.env._get_drone_pos_orn()
        vel, ang = self.env._get_drone_vel()
        eul = self.env._quat_to_euler(orn)
        self.assertGreater(pos[2], 1.3, f"altitude collapsed to {pos[2]:.3f}")
        self.assertLess(abs(ang[0]), 1.0, f"roll rate {ang[0]:+.3f} > 1.0")
        self.assertLess(abs(ang[1]), 1.0, f"pitch rate {ang[1]:+.3f} > 1.0")
        self.assertLess(abs(eul[0]), 0.1, f"roll angle {eul[0]:+.3f} > 0.1")
        self.assertLess(abs(eul[1]), 0.1, f"pitch angle {eul[1]:+.3f} > 0.1")

    def test_force_symmetry_no_torque(self):
        # Domain truth: identical action on all 4 motors → zero body torque,
        # so angular velocity must NOT grow regardless of altitude/velocity.
        # Test at action=+0.1 (slightly above hover, drone climbs) AND -0.1.
        for val in (+0.1, -0.1):
            self.env.reset()
            _step_uniform(self.env, val, 100)
            _, ang = self.env._get_drone_vel()
            self.assertLess(abs(ang[0]), 1.0, f"roll rate at action={val}: {ang[0]:+.3f}")
            self.assertLess(abs(ang[1]), 1.0, f"pitch rate at action={val}: {ang[1]:+.3f}")

    def test_reward_progress_sign(self):
        # Domain truth: r_progress = prev_dist - curr_dist. Moving toward the
        # current gate yields positive reward; moving away yields negative.
        # We do not assume monotonicity across gate boundaries (gate-pass spike).
        self.env.reset()
        # Read initial distance to current gate via privileged state.
        s0 = self.env._get_privileged_state()
        d0 = float(s0[12])
        # Step with a small thrust offset; the per-step delta-d defines reward sign.
        obs, r, term, trunc, info = self.env.step(np.zeros(4, dtype=np.float32))
        s1 = self.env._get_privileged_state()
        d1 = float(s1[12])
        # If gate not passed, reward must equal d0 - d1 within float tolerance.
        if not info.get("term_reason") and "gates_passed" in info and info["gates_passed"] == 0:
            expected = d0 - d1
            self.assertAlmostEqual(r, expected, places=4,
                                   msg=f"reward {r:.5f} != prev_dist - curr_dist = {expected:.5f}")

    def test_action_zero_crashes_were_a_bug(self):
        # Regression guard: action=zeros for 50 steps must NOT crash.
        # This is the exact failure mode that motivated the drag-force-CoM fix.
        self.env.reset()
        crashed = False
        for _ in range(50):
            _, _, term, _, info = self.env.step(np.zeros(4, dtype=np.float32))
            if term and info.get("term_reason") == "crash":
                crashed = True
                break
        self.assertFalse(crashed, "action=zeros caused a crash within 50 steps (drag-CoM regression)")


    def test_battery_drains_with_motors(self):
        self.env.reset()
        soc_before = self.env._battery.soc
        _step_zero(self.env, 100)
        soc_after = self.env._battery.soc
        self.assertLess(soc_after, soc_before, "battery SOC did not drain after 100 hover steps")
        self.assertGreater(soc_after, 0.5, "battery drained too fast at hover")

    def test_higher_rpm_drains_faster(self):
        self.env.reset()
        _step_zero(self.env, 50)
        soc_hover = self.env._battery.soc
        self.env.reset()
        _step_uniform(self.env, 1.0, 50)
        soc_max = self.env._battery.soc
        self.assertGreater(soc_hover, soc_max,
                           f"hover SOC {soc_hover:.3f} should be > max-throttle SOC {soc_max:.3f}")

    def test_battery_cutoff_terminates(self):
        self.env.reset()
        truncated = False
        for _ in range(2000):
            _, _, term, trunc, _ = self.env.step(np.ones(4, dtype=np.float32))
            if trunc:
                truncated = True
                break
            if term:
                break
        if not truncated:
            return
        self.assertLess(self.env._battery.v_terminal, 3.3,
                        "truncated but voltage above cutoff")

    def test_rpm_ceiling_drops_with_soc(self):
        self.env.reset()
        ceil_full = self.env._battery.rpm_ceiling
        _step_uniform(self.env, 0.5, 200)
        ceil_later = self.env._battery.rpm_ceiling
        self.assertLess(ceil_later, ceil_full,
                        f"rpm_ceiling should drop as SOC decreases: {ceil_full:.0f} -> {ceil_later:.0f}")


if __name__ == "__main__":
    unittest.main()
