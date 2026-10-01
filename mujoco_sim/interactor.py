"""Helpers to script interactor motions (pokes, presses, ...)."""

import mujoco
import numpy as np

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
END_PAUSE_SECONDS = 0.4
DEFAULT_STEP_DT = 0.002  # fallback when no model is available; match your sim step
# Smallest gap left between the claw's lowest geometry and the floor while
# pushing. Enough to stop the pads sinking into the ground without lifting the
# claw off the object it is pushing.
PUSH_FLOOR_CLEARANCE = 0.002


def pause_steps(seconds: float, dt: float) -> int:
    """Number of trajectory waypoints needed to hold for ``seconds``."""
    if seconds < 0:
        raise ValueError("pause duration cannot be negative")
    return int(round(seconds / dt))


def _lerp_segment(p0, p1, steps):
    if steps < 1:
        raise ValueError("steps must be at least 1")
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    if p0.shape != (3,) or p1.shape != (3,):
        raise ValueError("trajectory positions must be 3D coordinates")
    return [tuple(p0 + (p1 - p0) * t) for t in np.linspace(0, 1, steps)]


def poke_trajectory(
    start,
    target,
    approach_steps: int = 30,
    hold_steps: int = 20,
    retract_steps: int = 30,
    quat=IDENTITY_QUAT,
    end_pause_seconds: float = END_PAUSE_SECONDS,
    dt: float = DEFAULT_STEP_DT,
) -> list:
    """Waypoints for a single poke: approach -> hold at target -> retract -> pause."""
    if hold_steps < 0:
        raise ValueError("hold_steps cannot be negative")
    approach = _lerp_segment(start, target, approach_steps)
    hold = [tuple(np.asarray(target, dtype=float))] * hold_steps
    retract = _lerp_segment(target, start, retract_steps)
    trajectory = approach + hold + retract
    trajectory += [trajectory[-1]] * pause_steps(end_pause_seconds, dt)
    return [(p, quat) for p in trajectory]


def multi_poke_trajectory(
    waypoints: list,
    steps_per_segment: int = 30,
    quat=IDENTITY_QUAT,
    end_pause_steps: int = 0,
) -> list:
    """Interpolate a sequence of positions into scripted interactor poses."""
    if len(waypoints) < 2:
        raise ValueError("at least two waypoints are required")
    trajectory = []
    for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
        trajectory += _lerp_segment(p0, p1, steps_per_segment)
    trajectory += [trajectory[-1]] * end_pause_steps
    return [(p, quat) for p in trajectory]


def geom_local_half_extents(model, geom_id: int) -> np.ndarray:
    """Return the half-extents of one geom's bounding box in its own frame."""
    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return np.asarray(size, dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, size[0], dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.array([size[0], size[0], size[1] + size[0]], dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return np.array([size[0], size[0], size[1]], dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return np.asarray(size, dtype=float)
    return np.full(3, model.geom_rbound[geom_id], dtype=float)


def geom_world_extents(model, data, geom_id: int) -> np.ndarray:
    """Return world-axis half-extents of one geom's bounding box.

    Elongated shapes (the claw's capsule fingers, cylinders, ...) must use their
    own frame's extents so that thin axes are not inflated to the bounding
    sphere radius. Reusing ``geom_rbound`` on every axis used to widen the claw
    envelope by several centimetres, which made the proximity slowdown trigger
    while the claw was still visibly far from the object.
    """
    local = geom_local_half_extents(model, geom_id)
    return np.abs(data.geom_xmat[geom_id].reshape(3, 3)) @ local


def is_interactor_geom(model, geom_id: int) -> bool:
    """Whether ``geom_id`` belongs to a gripper rather than to the object.

    Identified by walking up to an ``interactor*`` body, not by looking for a
    mocap ancestor: a servo-driven claw (``scene._make_claw_dynamic``) is a
    plain finite-mass body, so the mocap test would misclassify every claw geom
    as object geometry.
    """
    body_id = model.geom_bodyid[geom_id]
    while body_id > 0:
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith(
            "interactor"
        ):
            return True
        body_id = model.body_parentid[body_id]
    return False


def object_geom_ids(model) -> list[int]:
    """Return every non-controller, non-floor geometry in the scene."""
    geom_ids = []
    for geom_id in range(model.ngeom):
        if model.geom(geom_id).name == "floor":
            continue
        if is_interactor_geom(model, geom_id):
            continue
        geom_ids.append(geom_id)
    return geom_ids


def object_bounds(model, data, body_name: str = "object") -> tuple[np.ndarray, np.ndarray]:
    """Return initial world-space bounds for the non-controller object geometry."""
    model.body(body_name)  # Validate the requested object name.
    points = list(data.flexvert_xpos) if model.nflexvert else []

    # Composite objects such as the rope can compile into top-level bodies.
    for geom_id in object_geom_ids(model):
        extents = geom_world_extents(model, data, geom_id)
        points.extend(
            [
                data.geom_xpos[geom_id] - extents,
                data.geom_xpos[geom_id] + extents,
            ]
        )

    if not points:
        raise ValueError(f"could not find geometry for body '{body_name}'")
    points = np.asarray(points, dtype=float)
    return points.min(axis=0), points.max(axis=0)


def controller_local_bounds(
    model, data, body_name: str = "interactor0"
) -> tuple[np.ndarray, np.ndarray]:
    """Return controller geometry bounds relative to its own origin.

    The origin is the controller body's resolved pose rather than its mocap
    reference, so this works for a servo-driven finite-mass claw as well as a
    mocap one (for a mocap body the two coincide). Proximity needs the geometry
    that is actually there, not the geometry that was ordered.
    """
    body = model.body(body_name)
    origin = data.xpos[body.id]
    points = []
    for geom_id in range(model.ngeom):
        ancestor = model.geom_bodyid[geom_id]
        while ancestor > 0 and ancestor != body.id:
            ancestor = model.body_parentid[ancestor]
        if ancestor != body.id:
            continue
        extents = geom_world_extents(model, data, geom_id)
        relative_center = data.geom_xpos[geom_id] - origin
        points.extend([relative_center - extents, relative_center + extents])
    if not points:
        raise ValueError(f"could not find geometry for controller '{body_name}'")
    points = np.asarray(points, dtype=float)
    return points.min(axis=0), points.max(axis=0)


def object_push_trajectory(
    model,
    data,
    steps_per_segment: int = 30,
    body_name: str = "object",
    margin: float = 0.06,
    approach_height: float = 0.16,
    controller_name: str = "interactor0",
    end_pause_seconds: float = END_PAUSE_SECONDS,
) -> list:
    """Approach, push through, and retract from an object's bounds.

    ``margin`` is the clearance between the controller geometry AABB and the
    object AABB, rather than the clearance between their reference points.
    """
    lower, upper = object_bounds(model, data, body_name)
    controller_lower, controller_upper = controller_local_bounds(
        model, data, controller_name
    )
    center = (lower + upper) * 0.5
    # The waypoint scripts the controller *origin*, but the pads hang below it
    # (29.3 mm at the open stance, which is where a push runs: the fingers are
    # never commanded). Scripting the origin at the object's centre therefore
    # drove the pads through the floor for the low objects - the rope sank
    # 17 mm and the heavy plank 4 mm - and turning the pad from a sphere into a
    # longer prism is what deepened it. Lift the origin only as far as it takes
    # for the lowest claw geometry to clear the ground; an object already
    # above that (every box) keeps its centre height untouched.
    contact_z = max(center[2], -controller_lower[2] + PUSH_FLOOR_CLEARANCE)
    side_x = lower[0] - controller_upper[0] - margin
    exit_x = upper[0] - controller_lower[0] + margin
    y = center[1]
    points = [
        (side_x, y, contact_z + approach_height),
        (side_x, y, contact_z),
        (exit_x, y, contact_z),
        (exit_x, y, contact_z + approach_height),
    ]
    return multi_poke_trajectory(
        points,
        steps_per_segment,
        end_pause_steps=pause_steps(end_pause_seconds, model.opt.timestep),
    )


def object_grip_lift_trajectory(
    model,
    data,
    steps_per_segment: int = 30,
    body_name: str = "object",
    approach_height: float = 0.14,
    lift_height: float = 0.16,
    end_pause_seconds: float = END_PAUSE_SECONDS,
) -> tuple[list, list[float]]:
    """Open one claw, approach the object, close its fingers, and lift.

    ``end_pause_seconds`` holds the claw on top after the lift.
    """
    body_id = model.body(body_name).id
    box_bounds = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        half_extents = model.geom_size[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_extents = np.abs(rotation) @ half_extents
        center = data.geom_xpos[geom_id]
        box_bounds.append((center - world_extents, center + world_extents))

    if box_bounds:
        lower = np.min([bounds[0] for bounds in box_bounds], axis=0)
        upper = np.max([bounds[1] for bounds in box_bounds], axis=0)
    else:
        # Soft and composite objects have no box geometry to grip; fall back to
        # the bounds of their flex vertices and remaining scene geometry.
        lower, upper = object_bounds(model, data, body_name)
    center = (lower + upper) * 0.5
    contact_z = max(0.012, center[2])
    # The claw's palm sits 0.148 m above its reference point (claw.xml): on an
    # object taller than the palm's reach, descending to the centre would drive
    # the palm into the object instead of the fingers closing around it, so
    # grip as low as the palm still clears the top. Shorter objects - every
    # existing case - keep their centre height.
    contact_z = max(contact_z, upper[2] - 0.148)
    y = center[1]
    approach_x = center[0]
    if not box_bounds and model.nflexvert:
        # Grip the middle of the body's cross-section rather than of its
        # bounding box: an asymmetric soft object (the sloth's waist sits well
        # off its own bounds centre) must be straddled exactly, otherwise the
        # open pads descend through one side of it.
        band = np.abs(data.flexvert_xpos[:, 1] - y) <= 0.02
        if band.sum() > 1:
            xs = data.flexvert_xpos[band, 0]
            approach_x = float((xs.min() + xs.max()) * 0.5)

    waypoints = [
        (approach_x, y, contact_z + approach_height),
        (approach_x, y, contact_z),
        (approach_x, y, contact_z),
        (approach_x, y, contact_z + lift_height),
    ]
    end_pause = pause_steps(end_pause_seconds, model.opt.timestep)
    trajectory = multi_poke_trajectory(
        waypoints, steps_per_segment, end_pause_steps=end_pause
    )
    closing = (
        [0.0] * steps_per_segment
        + list(np.linspace(0.0, 1.0, steps_per_segment))
        + [1.0] * steps_per_segment
        + [1.0] * end_pause  # keep the claw closed during the final pause
    )
    return trajectory, closing