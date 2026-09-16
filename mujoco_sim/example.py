"""Demo: the claw approaches from the side, sweeps through the object to
push it across the floor, then retracts. Shows the object actually moving
in response to the interaction (not just a poke-in-place)."""

import numpy as np
from PIL import Image

from interactor import multi_poke_trajectory
from scene import load_model
from simulation import DigitalTwinSim

sim = DigitalTwinSim(model=load_model(n_interactors=1), width=256, height=256)

# Waypoints (x, y, z) for the claw's contact point, in world coords.
# Object sits at origin, half-size 0.05, so we contact it at z=0.05 (mid-height).
waypoints = [
    (-0.20, 0.0, 0.20),   # start, high and to the side
    (-0.20, 0.0, 0.05),   # descend to push height
    (0.20, 0.0, 0.05),    # sweep through the object, pushing it along +x
    (0.20, 0.0, 0.20),    # lift back up
]
trajectory = multi_poke_trajectory(waypoints, steps_per_segment=150)

# `steps_per_segment` controls how finely the claw's motion is interpolated,
# and `substeps` is how many physics steps run per waypoint. Both need to be
# high enough that the claw doesn't jump further per step than its own
# contact geometry -- otherwise it tunnels/glances off the object instead of
# pushing it (too coarse and you'll see almost no displacement).
frames = sim.rollout(
    interactor_trajectory={"interactor0": trajectory},
    object_bodies=["object"],
    capture_every=8,
    substeps=4,
)

pos_start = frames[0].object_qpos["object"][:3]
pos_end = frames[-1].object_qpos["object"][:3]
print(f"collected {len(frames)} frames")
print(f"object moved from {pos_start.round(3)} to {pos_end.round(3)}")
print(f"displacement along x: {pos_end[0] - pos_start[0]:.3f} m")

# Save a few frames across the rollout so you can see the push happen.
sample_idxs = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
for i in sample_idxs:
    rgb, _ = frames[i].rgbd["cam_side"]
    Image.fromarray(rgb).save(f"./mujoco_sim/output/_frame_{i}.png")

# Also save an animated GIF of the whole rollout for a quick visual check.
gif_frames = [Image.fromarray(f.rgbd["cam_side"][0]) for f in frames]
gif_frames[0].save(
    "./mujoco_sim/output/push_demo.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=80,
    loop=0,
)