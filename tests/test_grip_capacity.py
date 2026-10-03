"""Grip capacity must exceed object weight, and must not scale with node mass.

Commit bc71404 ("add learned point mass anchors + global CMA mass") made contact
stiffness mass-normalized -- ``F = K * m_i * penetration`` -- so penetration depth
would not change when ``learn_mass`` altered a node's mass. That made total grip
force scale with the *mass sitting in the contact patch* rather than with the load
the pad applies:

    F_total = K * (sum of m_i over contacting nodes) * sum(penetration)

A rigid pad transmits whatever force the actuator applies; it does not care how
many surface samples were sprinkled on the object. For a gripper pinching a ball
with two small pad patches touching ~1.6% of the surface, sum(m_i) is ~0.4% of the
object mass, so grip collapsed to ~0.002x the object's weight and *no* value of mu
could lift it. The defect is invisible for broad contact (when nearly every node
touches, sum(m_i) -> M) and only appears for localized contact, which is exactly
what gripping is.

Penetration constants are measured on the real recorded grip data; see
BALL_* / SEAL_* below.
"""

import unittest

from qqtt.utils import cfg

GRAVITY = 9.81

# sim_soft_ball_grip_lift/final_data.pkl: 0.0434 m summed penetration from 19 of
# the 1200 tracked surface points, over a simulator carrying 5037 mass nodes
# (surface + interior fill). The case that broke.
BALL_TOTAL_NODES = 5037
BALL_SUM_PENETRATION = 0.0434

# sim_soft_seal_grip_lift: broad contact, most of the surface engaged.
SEAL_SUM_PENETRATION = 0.902


class GripCapacityTests(unittest.TestCase):
    """``cfg.controller_contact_stiffness`` is absolute (N/m), so grip capacity is
    ``mu * K * sum(penetration)`` regardless of the object's mass distribution."""

    def setUp(self):
        self.weight = cfg.object_total_mass * GRAVITY

    def grip_margin(self, sum_penetration):
        force = cfg.controller_contact_stiffness * sum_penetration
        return force / self.weight

    def test_localized_grip_can_lift_the_ball(self):
        """A small contact patch must still clear the object's weight.

        This is the case that broke: a 1.6% patch on a 0.3 kg ball.
        """
        margin = self.grip_margin(BALL_SUM_PENETRATION)
        self.assertGreater(
            margin,
            1.0,
            f"ball grip is {margin:.3f}x the {self.weight:.3f} N weight; "
            f"friction cannot lift the object",
        )

    def test_mass_normalized_law_would_fail(self):
        """Pin why the absolute law is required, so it is not silently reverted.

        Scales grip by the contact patch's share of the object mass. If this ever
        passes, the force law changed some other way and the numbers above need
        re-deriving.
        """
        node_mass = cfg.object_total_mass / BALL_TOTAL_NODES
        margin = cfg.controller_contact_stiffness * node_mass * BALL_SUM_PENETRATION / self.weight
        self.assertLess(
            margin,
            0.01,
            "mass-normalized contact unexpectedly holds the ball; "
            "re-derive these tests before trusting them",
        )

    def test_broad_grip_still_holds_for_every_case(self):
        """Absolute stiffness must clear weight for broad-contact cases too."""
        for name, sum_penetration in (
            ("seal", SEAL_SUM_PENETRATION),
            ("ball", BALL_SUM_PENETRATION),
        ):
            margin = cfg.controller_contact_friction * self.grip_margin(sum_penetration)
            self.assertGreater(
                margin, 1.0, f"{name}: friction margin {margin:.3f}x cannot lift"
            )

    def test_absolute_stiffness_is_explicit_integration_stable(self):
        """With F = K*pen the per-node acceleration is K*pen/m, so explicit Euler
        needs K <~ m*(0.5/dt)^2. The simulator enforces this too; assert it holds
        across the node counts the real cases use.

        Note the ceiling is mesh-density dependent (finer mesh -> lighter nodes ->
        a given K accelerates harder). Real cases sit at 1.2k-13k nodes, where
        K=2000 has headroom. Remesh much finer and lower K rather than raising
        this bound.
        """
        for n_nodes in (BALL_TOTAL_NODES, 11723):
            node_mass = cfg.object_total_mass / n_nodes
            k_max = node_mass * (0.5 / cfg.dt) ** 2
            self.assertLessEqual(
                cfg.controller_contact_stiffness,
                k_max,
                f"n={n_nodes}: contact stiffness {cfg.controller_contact_stiffness} "
                f"exceeds the explicit-integration limit {k_max:.1f} N/m",
            )


if __name__ == "__main__":
    unittest.main()