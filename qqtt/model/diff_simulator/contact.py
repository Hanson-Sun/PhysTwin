"""Shared tangential friction primitive for Warp contact kernels.

Velocity-level Coulomb friction: the friction force is not integrated as a
force but applied as a constraint on the relative velocity. For one contact the
constrained minimisation of the tangential relative velocity subject to a
friction cone has a closed form, so no iteration or global solve is needed.

Properties that matter for the mass-spring simulator:

* **Dissipative by construction.** The corrected relative velocity never has a
  larger tangential component than the input, so the law cannot inject energy.
  There is no spring constant, hence no ``dt * k / m`` stability condition and
  no ringing at the stick/slip transition.
* **Exact stiction.** Below the cone the whole tangential relative velocity is
  cancelled, so a held particle has exactly zero drift. Smoothing the cone is
  what destroys this: any blend that under-removes velocity has a nonzero
  steady-state creep fixed point.
* **Coulomb-exact in slip.** Above the cone exactly ``mu * N`` of force is
  applied, independent of the slip speed.
* **Stateless.** No stored displacement, so nothing persists on the autograd
  tape between substeps.
"""

import warp as wp


@wp.func
def coulomb_project(
    v_rel: wp.vec3,
    normal: wp.vec3,
    normal_load: float,
    mass: float,
    mu: float,
    dt: float,
):
    """Cancel or clamp the tangential relative velocity at a single contact.

    ``mu * normal_load * dt`` is the largest tangential impulse the cone allows,
    so the largest velocity change friction can impose is
    ``limit = mu * normal_load * dt / mass``.

    * ``|v_t| <= limit`` -- stick: the relative velocity becomes purely normal.
    * ``|v_t| > limit``  -- slip: remove exactly ``limit``.

    The normal component of ``v_rel`` is never modified, so this composes with
    the normal contact handling. ``mu = 0`` or ``normal_load = 0`` gives
    ``limit = 0`` and the function is the identity.
    """
    safe_mass = wp.max(mass, 1e-8)
    v_t = v_rel - wp.dot(v_rel, normal) * normal
    limit = mu * normal_load * dt / safe_mass
    speed = wp.length(v_t)
    if speed <= limit or speed < 1e-12:
        return v_rel - v_t
    return v_rel - v_t * (limit / speed)
