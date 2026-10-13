"""Regression checks for fixed-time trajectory extension."""
import unittest
import numpy as np
from Trajectory_Gen import (
    MANEUVER_STYLES, generate_single_style_trajectory,
    generate_aerobatic_trajectories_pvR,
)


class TrajectoryExtensionTests(unittest.TestCase):
    def test_longer_sequences_preserve_prefix_and_sampling_rate(self):
        for style in MANEUVER_STYLES:
            if style == 'half_moon':
                continue  # This finite arc intentionally spans the full sequence.
            with self.subTest(style=style):
                short = generate_single_style_trajectory(style, 80)
                long = generate_single_style_trajectory(style, 320)
                np.testing.assert_allclose(short, long[:80], atol=1e-12)
                fine = generate_single_style_trajectory(style, 640, delta_T=0.05)
                np.testing.assert_allclose(long, fine[::2], atol=1e-12)
                self.assertTrue(np.isfinite(long).all())

    def test_half_moon_always_spans_half_circle(self):
        for length in [2, 60, 160, 320, 1000]:
            xyz = generate_single_style_trajectory('half_moon', length, radius=5)
            angles = np.unwrap(np.arctan2(xyz[:, 1], xyz[:, 0]))
            np.testing.assert_allclose(angles, np.linspace(0, np.pi, length), atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(xyz[:, :2], axis=1), 5, atol=1e-12)

    def test_finite_maneuver_junctions_have_continuous_velocity(self):
        for style, junctions in [('split_s', [3.3, 6.0]),
                                 ('immelmann', [3.6, 6.0]), ('arch', [6.0]),
                                 ('spiral_inward', [6.0]), ('spiral_outward', [6.0]),
                                 ('conical_helix', [6.0])]:
            with self.subTest(style=style):
                dt = 0.0001
                xyz = generate_single_style_trajectory(style, 60003, delta_T=dt)
                for time in junctions:
                    k = round(time / dt)
                    before = (xyz[k] - xyz[k-1]) / dt
                    after = (xyz[k+1] - xyz[k]) / dt
                    np.testing.assert_allclose(before, after, atol=0.01, rtol=0)

    def test_radius_stays_positive_and_arch_tail_is_bounded(self):
        xyz = generate_single_style_trajectory('spiral_inward', 1000)
        r = np.linalg.norm(xyz[:, :2], axis=1)
        self.assertGreaterEqual(r.min(), 1 - 1e-10)
        np.testing.assert_allclose(r[60:], 1, atol=1e-10)
        arch = generate_single_style_trajectory('arch', 1000)
        self.assertGreaterEqual(arch[:, 2].min(), 6 - 1e-10)

    def test_generated_velocities_use_requested_time_step(self):
        np.random.seed(42)
        dt = 0.05
        states = generate_aerobatic_trajectories_pvR(30, 160, delta_T=dt).numpy()
        expected = np.gradient(states[:, :, 1:4], dt, axis=1)
        np.testing.assert_allclose(states[:, :, 4:7], expected, atol=2e-4, rtol=1e-4)
        self.assertTrue(np.isfinite(states).all())


if __name__ == '__main__':
    unittest.main()
