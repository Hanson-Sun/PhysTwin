"""Helpers to script interactor motions (pokes, presses, ...)."""

import mujoco
import numpy as np

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


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
) -> list:
    """Waypoints for a single poke: approach -> hold at target -> retract."""
    if hold_steps < 0:
        raise ValueError("hold_steps cannot be negative")
    approach = _lerp_segment(start, target, approach_steps)
    hold = [tuple(np.asarray(target, dtype=float))] * hold_steps
    retract = _lerp_segment(target, start, retract_steps)
    return [(p, quat) for p in approach + hold + retract]


def multi_poke_trajectory(waypoints: list, steps_per_segment: int = 30, quat=IDENTITY_QUAT) -> list:
    """Interpolate a sequence of positions into scripted interactor poses."""
    if len(waypoints) < 2:
        raise ValueError("at least two waypoints are required")
    trajectory = []
    for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
        trajectory += _lerp_segment(p0, p1, steps_per_segment)
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


def object_geom_ids(model) -> list[int]:
    """Return every non-controller, non-floor geometry in the scene."""
    geom_ids = []
    for geom_id in range(model.ngeom):
        if model.geom(geom_id).name == "floor":
            continue
        ancestor = model.geom_bodyid[geom_id]
        while ancestor > 0 and model.body(ancestor).mocapid[0] < 0:
            ancestor = model.body_parentid[ancestor]
        if ancestor > 0:
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
    """Return controller geometry bounds relative to its mocap origin."""
    body = model.body(body_name)
    mocap_id = body.mocapid[0]
    if mocap_id < 0:
        raise ValueError(f"'{body_name}' is not a mocap body")
    origin = data.mocap_pos[mocap_id]
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
    # Keep small floor-level objects (for example the rope) at their actual
    # center height instead of lifting the interactor above them.
    contact_z = max(0.012, center[2])
    side_x = lower[0] - controller_upper[0] - margin
    exit_x = upper[0] - controller_lower[0] + margin
    y = center[1]
    points = [
        (side_x, y, contact_z + approach_height),
        (side_x, y, contact_z),
        (exit_x, y, contact_z),
        (exit_x, y, contact_z + approach_height),
    ]
    return multi_poke_trajectory(points, steps_per_segment)


def object_grip_lift_trajectory(
    model,
    data,
    steps_per_segment: int = 30,
    body_name: str = "object",
    approach_height: float = 0.14,
    lift_height: float = 0.16,
) -> tuple[list, list[float]]:
    """Open one claw, approach the object, close its fingers, and lift."""
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
    trajectory = multi_poke_trajectory(waypoints, steps_per_segment)
    closing = (
        [0.0] * steps_per_segment
        + list(np.linspace(0.0, 1.0, steps_per_segment))
        + [1.0] * steps_per_segment
    )
    return trajectory, closing
