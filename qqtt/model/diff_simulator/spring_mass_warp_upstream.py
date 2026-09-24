import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


GROUND_GRAVITY = 9.81


def _fps_indices(points_np, K, seed=42):
    """Farthest-point sampling: returns K indices even over N points."""
    import numpy as np

    N = points_np.shape[0]
    if K <= 0 or K >= N:
        return np.arange(N, dtype=np.int32)
    rng = np.random.default_rng(int(seed))
    selected = np.empty(K, dtype=np.int32)
    first = int(rng.integers(0, N))
    selected[0] = first
    # min distance to selected set
    dist = np.linalg.norm(points_np - points_np[first], axis=1)
    for i in range(1, K):
        idx = int(np.argmax(dist))
        selected[i] = idx
        new_d = np.linalg.norm(points_np - points_np[idx], axis=1)
        dist = np.minimum(dist, new_d)
    return selected


def _knn_anchor_weights(points_np, anchor_points_np, knn=4):
    """kNN inverse-distance weights: N points x K anchors -> N*knn indices/weights."""
    import numpy as np
    import torch

    N = points_np.shape[0]
    K = anchor_points_np.shape[0]
    knn = max(1, min(int(knn), K))
    pts = torch.from_numpy(points_np.astype(np.float32))
    anc = torch.from_numpy(anchor_points_np.astype(np.float32))
    # N=700 K=128 -> 90k dists trivial on CPU
    dists = torch.cdist(pts, anc)  # N,K
    vals, idx = torch.topk(dists, k=knn, largest=False, dim=1)
    w = 1.0 / (vals + 1e-6)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return idx.numpy().astype(np.int32).reshape(-1), w.numpy().astype(np.float32).reshape(-1)


class State:
    def __init__(self, wp_init_vertices, num_control_points, num_contact_points=0):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)
        self.wp_contact_x = wp.zeros(
            (num_contact_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_contact_x_prev = wp.zeros_like(self.wp_contact_x, requires_grad=False)
        self.wp_contact_v = wp.zeros_like(self.wp_contact_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()

    # This takes more time but not necessary, will be overwritten directly
    # def clear_control(self):
    #     self.wp_control_x.zero_()
    #     self.wp_control_v.zero_()

    # def clear_states(self):
    #     self.wp_x.zero_()
    #     self.wp_v_before_ground.zero_()
    #     self.wp_v.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    substep_dt: float,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
):
    # Set the interpolated position and kinematic velocity for each substep.
    tid = wp.tid()

    displacement = target_control_point[tid] - original_control_point[tid]
    t = float(step + 1) / float(num_substeps)
    control_x[tid] = original_control_point[tid] + displacement * t
    control_v[tid] = displacement / (float(num_substeps) * substep_dt)


@wp.kernel(enable_backward=False)
def set_contact_points(
    num_substeps: int,
    substep_dt: float,
    original_contact_point: wp.array(dtype=wp.vec3),
    target_contact_point: wp.array(dtype=wp.vec3),
    step: int,
    contact_x: wp.array(dtype=wp.vec3),
    contact_x_prev: wp.array(dtype=wp.vec3),
    contact_v: wp.array(dtype=wp.vec3),
):
    # Store both ends of the controller segment swept during this substep.
    tid = wp.tid()
    displacement = target_contact_point[tid] - original_contact_point[tid]
    t_prev = float(step) / float(num_substeps)
    t = float(step + 1) / float(num_substeps)
    contact_x_prev[tid] = original_contact_point[tid] + displacement * t_prev
    contact_x[tid] = original_contact_point[tid] + displacement * t
    contact_v[tid] = displacement / (float(num_substeps) * substep_dt)


@wp.func
def closest_point_on_segment(point: wp.vec3, segment_start: wp.vec3, segment_end: wp.vec3):
    segment = segment_end - segment_start
    segment_length_squared = wp.dot(segment, segment)
    segment_t = wp.clamp(
        wp.dot(point - segment_start, segment) / wp.max(segment_length_squared, 1e-12),
        0.0,
        1.0,
    )
    return segment_start + segment_t * segment


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float),
    dashpot_damping: float,
    spring_Y_min: float,
    spring_Y_max: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if wp.exp(spring_Y[tid]) > spring_Y_min:

        idx1 = springs[tid][0]
        idx2 = springs[tid][1]

        if idx1 >= num_object_points:
            x1 = control_x[idx1 - num_object_points]
            v1 = control_v[idx1 - num_object_points]
        else:
            x1 = x[idx1]
            v1 = v[idx1]
        if idx2 >= num_object_points:
            x2 = control_x[idx2 - num_object_points]
            v2 = control_v[idx2 - num_object_points]
        else:
            x2 = x[idx2]
            v2 = v[idx2]

        rest = rest_lengths[tid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)

        spring_force = (
            wp.clamp(wp.exp(spring_Y[tid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )

        v_rel = wp.dot(v2 - v1, d)
        dashpot_forces = dashpot_damping * v_rel * d

        overall_force = spring_force + dashpot_forces

        if idx1 < num_object_points:
            wp.atomic_add(f, idx1, overall_force)
        if idx2 < num_object_points:
            wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def controller_contact_force(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    controller_x: wp.array(dtype=wp.vec3),
    controller_x_prev: wp.array(dtype=wp.vec3),
    controller_v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    controller_grid: wp.uint64,
    controller_sweep_radius: float,
    contact_radius: float,
    activation_radius: float,
    release_radius: float,
    contact_stiffness: float,
    contact_friction: float,
    contact_active: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    active_count: wp.array(dtype=wp.int32),
    f: wp.array(dtype=wp.vec3),
    normal_force_out: wp.array(dtype=wp.vec3),
    friction_force_out: wp.array(dtype=wp.vec3),
):
    """Apply unilateral normal contact and capped tangential friction.
    Contact stiffness is mass-normalized so penetration does not scale with
    object mass: F = (k_base * m_i) * pen, a = F/m_i = k_base*pen.
    With m_i=1 the behaviour is identical to the pre-mass-field baseline.
    """
    object_idx = wp.tid()
    object_position = x[object_idx]
    object_velocity = v[object_idx]
    # Mass-normalize to keep visual non-penetration independent of (learned) mass.
    # Base stiffness k_base was tuned for m=1, so effective k = k_base * m_i.
    mass_i = wp.max(masses[object_idx], 1e-6)
    stiffness = contact_stiffness * mass_i
    friction = contact_friction
    total_force = wp.vec3(0.0, 0.0, 0.0)
    total_normal_force = wp.vec3(0.0, 0.0, 0.0)
    total_friction_force = wp.vec3(0.0, 0.0, 0.0)
    was_active = contact_active[object_idx] != 0
    has_activation = int(0)
    has_release = int(0)
    neighbors = wp.hash_grid_query(
        controller_grid, object_position, release_radius + controller_sweep_radius
    )
    for controller_idx in neighbors:
        closest = closest_point_on_segment(
            object_position, controller_x_prev[controller_idx], controller_x[controller_idx]
        )
        delta = object_position - closest
        distance = wp.length(delta)
        if distance <= activation_radius:
            has_activation = 1
        if distance <= release_radius:
            has_release = 1

    is_active = (has_release != 0) if was_active else (has_activation != 0)
    contact_active[object_idx] = 1 if is_active else 0
    if is_active:
        wp.atomic_add(active_count, 0, 1)

    if is_active:
        neighbors = wp.hash_grid_query(
            controller_grid, object_position, contact_radius + controller_sweep_radius
        )
        for controller_idx in neighbors:
            closest = closest_point_on_segment(
                object_position, controller_x_prev[controller_idx], controller_x[controller_idx]
            )
            delta = object_position - closest
            distance = wp.length(delta)
            penetration = contact_radius - distance
            if penetration > 0.0:
                normal = delta / wp.max(distance, 1e-6)
                normal_force = stiffness * penetration
                normal_component = normal_force * normal
                total_normal_force += normal_component
                total_force += normal_component

                relative_velocity = object_velocity - controller_v[controller_idx]
                tangential_velocity = relative_velocity - wp.dot(relative_velocity, normal) * normal
                tangential_speed = wp.length(tangential_velocity)
                if tangential_speed > 1e-6:
                    friction_limit = friction * normal_force
                    requested_friction = tangential_speed * stiffness
                    friction_force = friction_limit * wp.tanh(
                        requested_friction / wp.max(friction_limit, 1e-8)
                    )
                    friction_component = -friction_force * tangential_velocity / tangential_speed
                    total_friction_force += friction_component
                    total_force += friction_component
                wp.atomic_add(contact_count, 0, 1)
    normal_force_out[object_idx] = total_normal_force
    friction_force_out[object_idx] = total_friction_force
    wp.atomic_add(f, object_idx, total_force)


@wp.kernel
def controller_contact_force_fixed(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    controller_x: wp.array(dtype=wp.vec3),
    controller_x_prev: wp.array(dtype=wp.vec3),
    controller_v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    controller_grid: wp.uint64,
    controller_sweep_radius: float,
    contact_radius: float,
    activation_radius: float,
    release_radius: float,
    contact_stiffness: float,
    contact_friction: float,
    contact_active: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    active_count: wp.array(dtype=wp.int32),
    f: wp.array(dtype=wp.vec3),
):
    """Apply fixed contact for CMA without calibration component buffers.
    Mass-normalized like the learnable variant.
    """
    object_idx = wp.tid()
    object_position = x[object_idx]
    object_velocity = v[object_idx]
    # Mass-normalize: effective stiffness scales with per-vertex mass so that
    # penetration depth `pen = m*a/k_eff` stays independent of global/local mass.
    _mass_i = wp.max(masses[object_idx], 1e-6)
    _stiffness_eff = contact_stiffness * _mass_i
    # Shadow names used below so the remainder of the kernel is unchanged.
    contact_stiffness = _stiffness_eff
    total_force = wp.vec3(0.0, 0.0, 0.0)
    was_active = contact_active[object_idx] != 0
    has_activation = int(0)
    has_release = int(0)
    neighbors = wp.hash_grid_query(
        controller_grid, object_position, release_radius + controller_sweep_radius
    )
    for controller_idx in neighbors:
        closest = closest_point_on_segment(
            object_position, controller_x_prev[controller_idx], controller_x[controller_idx]
        )
        delta = object_position - closest
        distance = wp.length(delta)
        if distance <= activation_radius:
            has_activation = 1
        if distance <= release_radius:
            has_release = 1

    is_active = (has_release != 0) if was_active else (has_activation != 0)
    contact_active[object_idx] = 1 if is_active else 0
    if is_active:
        wp.atomic_add(active_count, 0, 1)

    if is_active:
        neighbors = wp.hash_grid_query(
            controller_grid, object_position, contact_radius + controller_sweep_radius
        )
        for controller_idx in neighbors:
            closest = closest_point_on_segment(
                object_position, controller_x_prev[controller_idx], controller_x[controller_idx]
            )
            delta = object_position - closest
            distance = wp.length(delta)
            penetration = contact_radius - distance
            if penetration > 0.0:
                normal = delta / wp.max(distance, 1e-6)
                normal_force = contact_stiffness * penetration
                total_force += normal_force * normal

                relative_velocity = object_velocity - controller_v[controller_idx]
                tangential_velocity = relative_velocity - wp.dot(relative_velocity, normal) * normal
                tangential_speed = wp.length(tangential_velocity)
                if tangential_speed > 1e-6:
                    friction_limit = contact_friction * normal_force
                    requested_friction = tangential_speed * contact_stiffness
                    friction_force = friction_limit * wp.tanh(
                        requested_friction / wp.max(friction_limit, 1e-8)
                    )
                    total_force -= friction_force * tangential_velocity / tangential_speed
                wp.atomic_add(contact_count, 0, 1)
    wp.atomic_add(f, object_idx, total_force)


@wp.kernel
def apply_controller_contact_calibration(
    normal_force: wp.array(dtype=wp.vec3),
    friction_force: wp.array(dtype=wp.vec3),
    contact_stiffness: wp.array(dtype=wp.float32),
    contact_friction: wp.array(dtype=wp.float32),
    base_stiffness: float,
    base_friction: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    stiffness_scale = wp.exp(contact_stiffness[0]) / base_stiffness
    friction_scale = wp.clamp(contact_friction[0], low=0.0, high=2.0) / wp.max(base_friction, 1e-6)
    correction = (stiffness_scale - 1.0) * normal_force[tid]
    correction += (friction_scale - 1.0) * friction_force[tid]
    wp.atomic_add(f, tid, correction)


@wp.kernel
def compute_masses_from_log(
    log_mass: wp.array(dtype=wp.float32),
    mass_min: float,
    mass_max: float,
    masses: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    m = wp.exp(log_mass[tid])
    masses[tid] = wp.clamp(m, low=mass_min, high=mass_max)


@wp.kernel
def compute_masses_from_anchor_log(
    anchor_log_mass: wp.array(dtype=wp.float32),
    anchor_indices: wp.array(dtype=wp.int32),
    anchor_weights: wp.array(dtype=wp.float32),
    knn: int,
    mass_min: float,
    mass_max: float,
    masses: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    log_m = float(0.0)
    base = tid * knn
    # Unrolled up to 8 neighbours; knn is typically 4.
    for k in range(8):
        if k < knn:
            idx = anchor_indices[base + k]
            w = anchor_weights[base + k]
            log_m += w * anchor_log_mass[idx]
    masses[tid] = wp.clamp(wp.exp(log_m), low=mass_min, high=mass_max)


@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[tid]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
):
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            friction_ratio = (
                clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length
            )
            # Smooth only the Coulomb impulse cap; collision detection and
            # impact timing remain hard-thresholded below.
            a = 1.0 - wp.tanh(friction_ratio)
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    mask1 = masks[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index != i:
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v1 = v[tid]
    m1 = masses[tid]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


@wp.kernel
def ground_friction_force(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    contact_smoothing: float,
    forces: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    position = x[tid]
    velocity = v[tid]
    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor
    next_signed_z = (position[2] + velocity[2] * dt) * reverse_factor
    # Keep the activation band narrow: points farther than four smoothing
    # widths from the plane still receive exactly zero ground friction.
    if next_signed_z <= 4.0 * contact_smoothing:
        contact_weight = 1.0 / (
            1.0
            + wp.exp(
                (next_signed_z - 4.0 * contact_smoothing) / contact_smoothing
            )
        )
        tangent_velocity = velocity - wp.dot(velocity, normal) * normal
        tangent_speed = wp.length(tangent_velocity)
        mass = wp.max(masses[tid], 1e-6)
        penetration = wp.max(-next_signed_z, 0.0)
        normal_force = contact_weight * mass * (
            GROUND_GRAVITY + penetration / wp.max(dt * dt, 1e-12)
        )
        friction_limit = wp.clamp(collide_fric[0], low=0.0, high=2.0) * normal_force
        requested_friction = mass * tangent_speed / wp.max(dt, 1e-6)
        # Smoothly saturate at mu*N while preserving the low-speed stopping
        # force. This replaces the nondifferentiable min(mu*N, m*v/dt).
        friction_force = friction_limit * wp.tanh(
            requested_friction / wp.max(friction_limit, 1e-8)
        )
        if tangent_speed > 1e-8:
            wp.atomic_sub(
                forces,
                tid,
                friction_force * tangent_velocity / tangent_speed,
            )


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    position = x[tid]
    velocity = v[tid]
    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor
    signed_z = position[2] * reverse_factor
    signed_v_z = velocity[2] * reverse_factor
    next_signed_z = (position[2] + velocity[2] * dt) * reverse_factor

    impact = next_signed_z < 0.0 and signed_v_z < -1e-4
    if impact:
        normal_velocity = wp.dot(velocity, normal) * normal
        tangent_velocity = velocity - normal_velocity
        elasticity = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        velocity_after = -elasticity * normal_velocity + tangent_velocity
        toi = wp.clamp(-signed_z / signed_v_z, low=0.0, high=dt)
    else:
        velocity_after = velocity
        toi = 0.0

    x_new[tid] = position + velocity * toi + velocity_after * (dt - toi)
    v_new[tid] = velocity_after


@wp.kernel(enable_backward=False)
def compute_distances(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    distances: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if gt_mask[i] == 1:
        dist = wp.length(gt[i] - pred[j])
        distances[i, j] = dist
    else:
        distances[i, j] = 1e6


@wp.kernel(enable_backward=False)
def compute_neigh_indices(
    distances: wp.array2d(dtype=float),
    neigh_indices: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_dist = float(1e6)
    min_index = int(-1)
    for j in range(distances.shape[1]):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            min_index = j
    neigh_indices[i] = min_index


@wp.kernel
def compute_chamfer_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    neigh_indices: wp.array(dtype=wp.int32),
    loss_weight: float,
    chamfer_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        min_pred = pred[neigh_indices[i]]
        min_dist = wp.length(min_pred - gt[i])
        final_min_dist = loss_weight * min_dist * min_dist / float(num_valid)
        wp.atomic_add(chamfer_loss, 0, final_min_dist)


@wp.kernel
def compute_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    loss_weight: float,
    track_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        pred_x = pred[i][0]
        pred_y = pred[i][1]
        pred_z = pred[i][2]
        gt_x = gt[i][0]
        gt_y = gt[i][1]
        gt_z = gt[i][2]

        dist_x = wp.abs(pred_x - gt_x)
        dist_y = wp.abs(pred_y - gt_y)
        dist_z = wp.abs(pred_z - gt_z)

        if dist_x < 1.0:
            temp_track_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_track_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_track_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_track_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_track_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_track_loss_z = dist_z - 0.5

        temp_track_loss = temp_track_loss_x + temp_track_loss_y + temp_track_loss_z

        average_factor = float(num_valid) * 3.0

        final_track_loss = loss_weight * temp_track_loss / average_factor

        wp.atomic_add(track_loss, 0, final_track_loss)


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    output[0] = input


@wp.kernel(enable_backward=False)
def update_acc(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    prev_acc[tid] = v2[tid] - v1[tid]


@wp.kernel
def compute_acc_loss(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
    num_object_points: int,
    acc_count: wp.array(dtype=wp.int32),
    acc_weight: float,
    acc_loss: wp.array(dtype=wp.float32),
):
    if acc_count[0] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        tid = wp.tid()
        cur_acc = v2[tid] - v1[tid]
        cur_x = cur_acc[0]
        cur_y = cur_acc[1]
        cur_z = cur_acc[2]

        prev_x = prev_acc[tid][0]
        prev_y = prev_acc[tid][1]
        prev_z = prev_acc[tid][2]

        dist_x = wp.abs(cur_x - prev_x)
        dist_y = wp.abs(cur_y - prev_y)
        dist_z = wp.abs(cur_z - prev_z)

        if dist_x < 1.0:
            temp_acc_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_acc_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_acc_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_acc_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_acc_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_acc_loss_z = dist_z - 0.5

        temp_acc_loss = temp_acc_loss_x + temp_acc_loss_y + temp_acc_loss_z

        average_factor = float(num_object_points) * 3.0

        final_acc_loss = acc_weight * temp_acc_loss / average_factor

        wp.atomic_add(acc_loss, 0, final_acc_loss)


@wp.kernel
def compute_final_loss(
    chamfer_loss: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    loss[0] = chamfer_loss[0] + track_loss[0] + acc_loss[0]


@wp.kernel
def compute_simple_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    num_object_points: int,
    loss: wp.array(dtype=wp.float32),
):
    # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
    tid = wp.tid()
    pred_x = pred[tid][0]
    pred_y = pred[tid][1]
    pred_z = pred[tid][2]

    gt_x = gt[tid][0]
    gt_y = gt[tid][1]
    gt_z = gt[tid][2]

    dist_x = wp.abs(pred_x - gt_x)
    dist_y = wp.abs(pred_y - gt_y)
    dist_z = wp.abs(pred_z - gt_z)

    if dist_x < 1.0:
        temp_simple_loss_x = 0.5 * (dist_x**2.0)
    else:
        temp_simple_loss_x = dist_x - 0.5

    if dist_y < 1.0:
        temp_simple_loss_y = 0.5 * (dist_y**2.0)
    else:
        temp_simple_loss_y = dist_y - 0.5

    if dist_z < 1.0:
        temp_simple_loss_z = 0.5 * (dist_z**2.0)
    else:
        temp_simple_loss_z = dist_z - 0.5

    temp_simple_loss = temp_simple_loss_x + temp_simple_loss_y + temp_simple_loss_z

    average_factor = float(num_object_points) * 3.0

    final_simple_loss = temp_simple_loss / average_factor

    wp.atomic_add(loss, 0, final_simple_loss)


class SpringMassSystemWarp:
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        controller_contact_points=None,
        controller_contact_stiffness=None,
        controller_contact_friction=None,
        learn_controller_contact=False,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        disable_backward=False,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device
        self.disable_backward = disable_backward
        self.ground_contact_smoothing = float(
            getattr(cfg, "ground_contact_smoothing", 2e-5)
        )

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        self.controller_points = controller_points
        self.controller_contact_points = (
            controller_contact_points
            if controller_contact_points is not None
            else controller_points
        )
        self.num_contact_points = (
            self.controller_contact_points.shape[1]
            if self.controller_contact_points is not None
            else 0
        )
        self.controller_contact_radius = float(getattr(cfg, "controller_contact_radius", 0.014))
        self.controller_contact_activation_radius = float(
            getattr(cfg, "controller_contact_activation_radius", self.controller_contact_radius)
        )
        self.controller_contact_release_radius = float(
            getattr(cfg, "controller_contact_release_radius", self.controller_contact_radius * 1.25)
        )
        if self.controller_contact_points is not None and self.controller_contact_points.shape[0] > 1:
            contact_motion = self.controller_contact_points[1:] - self.controller_contact_points[:-1]
            self.controller_contact_sweep_radius = float(
                torch.linalg.vector_norm(contact_motion, dim=-1).amax().item()
                / max(self.num_substeps, 1)
            )
        else:
            self.controller_contact_sweep_radius = 0.0
        if self.controller_contact_activation_radius > self.controller_contact_release_radius:
            raise ValueError("controller contact activation radius must not exceed release radius")
        if controller_contact_stiffness is None:
            controller_contact_stiffness = getattr(cfg, "controller_contact_stiffness", 3e4)
        if controller_contact_stiffness <= 0.0:
            raise ValueError("controller contact stiffness must be positive")
        if controller_contact_friction is None:
            controller_contact_friction = getattr(cfg, "controller_contact_friction", 0.3)
        self.controller_contact_stiffness = float(controller_contact_stiffness)
        self.controller_contact_friction = float(controller_contact_friction)
        self.learn_controller_contact = learn_controller_contact
        self.wp_controller_contact_stiffness = wp.from_torch(
            torch.log(
                torch.tensor(
                    [controller_contact_stiffness],
                    dtype=torch.float32,
                    device=self.device,
                )
            ),
            requires_grad=learn_controller_contact,
        )
        self.wp_controller_contact_friction = wp.from_torch(
            torch.tensor([controller_contact_friction], dtype=torch.float32, device=self.device),
            requires_grad=learn_controller_contact,
        )
        self.controller_contact_enabled = self.controller_contact_points is not None
        self.controller_contact_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)
        self.controller_active_contact_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)
        self.controller_contact_active = wp.zeros(
            self.num_object_points, dtype=wp.int32, requires_grad=False
        )
        if self.learn_controller_contact:
            self.controller_contact_normal_force = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
            self.controller_contact_friction_force = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.controller_contact_normal_force = None
            self.controller_contact_friction_force = None
        self.controller_contact_grid = (
            wp.HashGrid(128, 128, 128) if self.controller_contact_enabled else None
        )

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # # Do some initialization to initialize the warp cuda graph
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        # --- Learnable mass field (per-vertex or anchored FPS) ---
        self.learn_mass = bool(getattr(cfg, "learn_mass", False))
        self.mass_min = float(getattr(cfg, "mass_min", 0.1))
        self.mass_max = float(getattr(cfg, "mass_max", 10.0))
        self.init_mass = float(getattr(cfg, "init_mass", 1.0))
        if self.mass_min <= 0.0 or self.mass_max <= self.mass_min:
            raise ValueError(f"invalid mass bounds min={self.mass_min} max={self.mass_max}")
        # Anchored mode: K anchors via FPS/random, kNN interpolation to N points
        self.use_anchor = False
        self.num_anchors = 0
        self.anchor_knn = 0
        self.anchor_indices = None
        self.wp_anchor_indices = None
        self.wp_anchor_weights = None
        if self.learn_mass:
            base_masses = init_masses[:num_object_points].to(dtype=torch.float32, device=self.device)
            K = int(getattr(cfg, "mass_num_anchors", 0) or 0)
            knn = int(getattr(cfg, "mass_anchor_knn", 4) or 4)
            method = str(getattr(cfg, "mass_anchor_method", "fps") or "fps").lower()
            seed = int(getattr(cfg, "mass_anchor_seed", 42) or 42)
            N = int(num_object_points)
            use_anchor = 0 < K < N and method in ("fps", "random")
            if use_anchor:
                import numpy as np
                pts_np = init_vertices[:N].detach().cpu().numpy().astype(np.float32)
                if method == "fps":
                    anchor_idx = _fps_indices(pts_np, K, seed=seed)
                else:
                    rng = np.random.default_rng(int(seed))
                    anchor_idx = rng.choice(N, size=K, replace=False).astype(np.int32)
                    anchor_idx.sort()  # deterministic order
                anchor_pts = pts_np[anchor_idx]
                knn_clamped = max(1, min(int(knn), K))
                knn_idx, knn_w = _knn_anchor_weights(pts_np, anchor_pts, knn=knn_clamped)
                self.use_anchor = True
                self.num_anchors = int(K)
                self.anchor_knn = int(knn_clamped)
                self.anchor_indices = anchor_idx.astype(np.int32)
                self.wp_anchor_indices = wp.from_torch(
                    torch.from_numpy(knn_idx.astype(np.int32)).to(self.device),
                    dtype=wp.int32,
                    requires_grad=False,
                )
                self.wp_anchor_weights = wp.from_torch(
                    torch.from_numpy(knn_w.astype(np.float32)).to(self.device),
                    dtype=wp.float32,
                    requires_grad=False,
                )
                anchor_base = base_masses[self.anchor_indices].to(dtype=torch.float32, device=self.device)
                scaled = (anchor_base * self.init_mass).clamp(min=self.mass_min, max=self.mass_max).clamp(min=1e-6)
                init_log_mass = torch.log(scaled)
                self.wp_log_mass = wp.from_torch(init_log_mass, dtype=wp.float32, requires_grad=not disable_backward)
                self.wp_masses = wp.zeros(self.num_object_points, dtype=wp.float32, requires_grad=not disable_backward)
                wp.launch(
                    kernel=compute_masses_from_anchor_log,
                    dim=self.num_object_points,
                    inputs=[self.wp_log_mass, self.wp_anchor_indices, self.wp_anchor_weights, self.anchor_knn, self.mass_min, self.mass_max],
                    outputs=[self.wp_masses],
                )
                logger.info(f"[SIMULATION]: Anchored mass field K={K} knn={knn_clamped} method={method} seed={seed}")
            else:
                # Fallback: per-vertex field (N params)
                scaled = (base_masses * self.init_mass).clamp(min=self.mass_min, max=self.mass_max).clamp(min=1e-6)
                init_log_mass = torch.log(scaled)
                self.wp_log_mass = wp.from_torch(init_log_mass, dtype=wp.float32, requires_grad=not disable_backward)
                self.wp_masses = wp.zeros(self.num_object_points, dtype=wp.float32, requires_grad=not disable_backward)
                wp.launch(
                    kernel=compute_masses_from_log,
                    dim=self.num_object_points,
                    inputs=[self.wp_log_mass, self.mass_min, self.mass_max],
                    outputs=[self.wp_masses],
                )
        else:
            self.wp_log_mass = None
            base_masses = init_masses[:num_object_points].to(dtype=torch.float32, device=self.device)
            scaled = (base_masses * self.init_mass).clamp(min=1e-6)
            self.wp_masses = wp.from_torch(scaled.contiguous(), dtype=wp.float32, requires_grad=False)

        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_original_contact_point = wp.from_torch(
                self.controller_contact_points[0].clone(),
                dtype=wp.vec3,
                requires_grad=False,
            )
            self.wp_target_contact_point = wp.from_torch(
                self.controller_contact_points[1].clone(),
                dtype=wp.vec3,
                requires_grad=False,
            )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(
                self.wp_init_velocities,
                self.num_control_points,
                self.num_contact_points,
            )
            self.wp_states.append(state)
        if cfg.data_type == "real":
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), dtype=wp.int32, requires_grad=False
            )

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Create the CUDA graph to accelerate.
        if cfg.use_graph:
            self.rebuild_graphs()
        else:
            self.tape = wp.Tape()

    def rebuild_graphs(self):
        """Re-capture training and forward graphs after an eager replay."""
        if not cfg.use_graph:
            self.tape = wp.Tape()
            return

        if cfg.data_type == "real":
            if not self.disable_backward:
                with wp.ScopedCapture() as capture:
                    self.tape = wp.Tape()
                    with self.tape:
                        self.step()
                        self.calculate_loss()
                    self.tape.backward(self.loss)
            else:
                with wp.ScopedCapture() as capture:
                    self.step()
                    self.calculate_loss()
        elif cfg.data_type == "synthetic":
            if not self.disable_backward:
                with wp.ScopedCapture() as capture:
                    self.tape = wp.Tape()
                    with self.tape:
                        self.step()
                        self.calculate_simple_loss()
                    self.tape.backward(self.loss)
            else:
                with wp.ScopedCapture() as capture:
                    self.step()
                    self.calculate_simple_loss()
        else:
            raise NotImplementedError

        self.graph = capture.graph
        with wp.ScopedCapture() as forward_capture:
            self.step()
        self.forward_graph = forward_capture.graph

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_contact_points,
                inputs=[self.controller_contact_points[frame_idx - 1]],
                outputs=[self.wp_original_contact_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_contact_points,
                inputs=[self.controller_contact_points[frame_idx]],
                outputs=[self.wp_target_contact_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points == wp_x.shape[0]
            and self.num_object_points == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def update_collision_graph(self):
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def _refresh_masses(self):
        if self.learn_mass and self.wp_log_mass is not None:
            if getattr(self, "use_anchor", False):
                wp.launch(
                    kernel=compute_masses_from_anchor_log,
                    dim=self.num_object_points,
                    inputs=[self.wp_log_mass, self.wp_anchor_indices, self.wp_anchor_weights, self.anchor_knn, self.mass_min, self.mass_max],
                    outputs=[self.wp_masses],
                )
            else:
                wp.launch(
                    kernel=compute_masses_from_log,
                    dim=self.num_object_points,
                    inputs=[self.wp_log_mass, self.mass_min, self.mass_max],
                    outputs=[self.wp_masses],
                )

    def step(self):
        if self.learn_mass and self.wp_log_mass is not None:
            if getattr(self, "use_anchor", False):
                wp.launch(
                    kernel=compute_masses_from_anchor_log,
                    dim=self.num_object_points,
                    inputs=[self.wp_log_mass, self.wp_anchor_indices, self.wp_anchor_weights, self.anchor_knn, self.mass_min, self.mass_max],
                    outputs=[self.wp_masses],
                )
            else:
                wp.launch(
                    kernel=compute_masses_from_log,
                    dim=self.num_object_points,
                    inputs=[self.wp_log_mass, self.mass_min, self.mass_max],
                    outputs=[self.wp_masses],
                )
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()
            if not self.controller_points is None:
                # Set the legacy sparse control points.
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.dt,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[
                        self.wp_states[i].wp_control_x,
                        self.wp_states[i].wp_control_v,
                    ],
                )
                if self.num_contact_points:
                    wp.launch(
                        set_contact_points,
                        dim=self.num_contact_points,
                        inputs=[
                            self.num_substeps,
                            self.dt,
                            self.wp_original_contact_point,
                            self.wp_target_contact_point,
                            i,
                        ],
                        outputs=[
                            self.wp_states[i].wp_contact_x,
                            self.wp_states[i].wp_contact_x_prev,
                            self.wp_states[i].wp_contact_v,
                        ],
                    )

            # Calculate the spring forces
            wp.launch(
                kernel=eval_springs,
                dim=self.n_springs,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.num_object_points,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.dashpot_damping,
                    self.spring_Y_min,
                    self.spring_Y_max,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.controller_contact_enabled:
                self.controller_contact_grid.build(
                    self.wp_states[i].wp_contact_x,
                    self.controller_contact_release_radius,
                )
                self.controller_contact_count.zero_()
                self.controller_active_contact_count.zero_()
                contact_kernel = (
                    controller_contact_force
                    if self.learn_controller_contact
                    else controller_contact_force_fixed
                )
                contact_inputs = [
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_contact_x,
                    self.wp_states[i].wp_contact_x_prev,
                    self.wp_states[i].wp_contact_v,
                    self.wp_masses,
                    self.controller_contact_grid.id,
                    self.controller_contact_sweep_radius,
                    self.controller_contact_radius,
                    self.controller_contact_activation_radius,
                    self.controller_contact_release_radius,
                    self.controller_contact_stiffness,
                    self.controller_contact_friction,
                    self.controller_contact_active,
                    self.controller_contact_count,
                    self.controller_active_contact_count,
                ]
                if self.learn_controller_contact:
                    wp.launch(
                        contact_kernel,
                        dim=self.num_object_points,
                        inputs=contact_inputs,
                        outputs=[
                            self.wp_states[i].wp_vertice_forces,
                            self.controller_contact_normal_force,
                            self.controller_contact_friction_force,
                        ],
                    )
                    wp.launch(
                        apply_controller_contact_calibration,
                        dim=self.num_object_points,
                        inputs=[
                            self.controller_contact_normal_force,
                            self.controller_contact_friction_force,
                            self.wp_controller_contact_stiffness,
                            self.wp_controller_contact_friction,
                            self.controller_contact_stiffness,
                            self.controller_contact_friction,
                        ],
                        outputs=[self.wp_states[i].wp_vertice_forces],
                    )
                else:
                    wp.launch(
                        contact_kernel,
                        dim=self.num_object_points,
                        inputs=contact_inputs,
                        outputs=[self.wp_states[i].wp_vertice_forces],
                    )

            wp.launch(
                ground_friction_force,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_masses,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                    self.ground_contact_smoothing,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        # Compute the chamfer loss
        # Precompute the distances matrix for the chamfer loss
        wp.launch(
            compute_distances,
            dim=(self.num_original_points, self.num_surface_points),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_valid_visibilities,
                self.neigh_indices,
                cfg.chamfer_weight,
            ],
            outputs=[self.chamfer_loss],
        )

        # Compute the tracking loss
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_motions_valid,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )

    def clear_loss(self):
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
        self.loss.zero_()

    def set_controller_contact(self, contact_stiffness, contact_friction):
        stiffness = wp.from_torch(
            torch.log(contact_stiffness).to(self.device).contiguous(),
            dtype=wp.float32,
            requires_grad=False,
        )
        friction = wp.from_torch(
            contact_friction.to(self.device).contiguous(),
            dtype=wp.float32,
            requires_grad=False,
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[stiffness],
            outputs=[self.wp_controller_contact_stiffness],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[friction],
            outputs=[self.wp_controller_contact_friction],
        )

    def get_mass(self):
        """Return current per-vertex masses as a detached torch tensor (num_object_points, on cfg.device)."""
        if self.learn_mass and self.wp_log_mass is not None:
            # Ensure wp_masses is up to date on device before reading (eager, outside tape).
            self._refresh_masses()
        t = wp.to_torch(self.wp_masses, requires_grad=False).detach()
        # Return on original device (cuda) for training code that expects cuda tensors
        return t.clone()

    def get_mass_cpu(self):
        return self.get_mass().cpu()

    def get_log_mass(self):
        if self.wp_log_mass is None:
            return None
        return wp.to_torch(self.wp_log_mass, requires_grad=False).detach().clone()

    def get_anchor_indices(self):
        return None if not getattr(self, "use_anchor", False) else getattr(self, "anchor_indices", None)

    def set_log_mass(self, log_mass):
        assert self.wp_log_mass is not None, "learn_mass is disabled"
        log_t = log_mass.to(self.device, dtype=torch.float32).contiguous()
        if getattr(self, "use_anchor", False):
            assert log_t.numel() == int(self.num_anchors), f"expected {self.num_anchors} anchors, got {log_t.numel()}"
            wp_log = wp.from_torch(log_t, dtype=wp.float32, requires_grad=False)
            wp.launch(copy_float, dim=self.num_anchors, inputs=[wp_log], outputs=[self.wp_log_mass])
        else:
            assert log_t.numel() == self.num_object_points
            wp_log = wp.from_torch(log_t, dtype=wp.float32, requires_grad=False)
            wp.launch(copy_float, dim=self.num_object_points, inputs=[wp_log], outputs=[self.wp_log_mass])
        self._refresh_masses()

    def set_mass(self, masses):
        """Set masses via clamped log.
        In anchored mode `masses` is per-anchor (K,). Otherwise per-vertex (N,).
        """
        assert self.wp_log_mass is not None
        if getattr(self, "use_anchor", False):
            assert masses.numel() == int(self.num_anchors)
        m = masses.to(self.device, dtype=torch.float32).clamp(min=self.mass_min, max=self.mass_max).clamp(min=1e-6)
        self.set_log_mass(torch.log(m))

    # Functions used to load the parmeters
    def set_spring_Y(self, spring_Y):
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )

    def set_init_mass_scale(self, scale: float):
        """Rescale all masses by `scale` (used by CMA global-mass search when learn_mass is False)."""
        if self.learn_mass:
            raise RuntimeError("set_init_mass_scale is for non-learnable mass mode; use set_log_mass instead")
        cur = wp.to_torch(self.wp_masses, requires_grad=False).detach()
        new = (cur * float(scale)).clamp(min=1e-6)
        wp_new = wp.from_torch(new.contiguous(), dtype=wp.float32, requires_grad=False)
        wp.launch(copy_float, dim=self.num_object_points, inputs=[wp_new], outputs=[self.wp_masses])
