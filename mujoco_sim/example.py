"""Demo: push the default validated box across the floor with a scripted claw."""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .interactor import multi_poke_trajectory
    from .scene import load_model
    from .simulation import DigitalTwinSim
except ImportError:  # Support running this file directly from the repository root.
    from interactor import multi_poke_trajectory
    from scene import load_model
    from simulation import DigitalTwinSim


OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    sim = DigitalTwinSim(model=load_model(n_interactors=1), width=256, height=256)

    # Waypoints (x, y, z) for the claw's contact point, in world coordinates.
    # The default box sits at the origin; z=0.05 reaches its center contact
    # plane.
    waypoints = [
        (-0.20, 0.0, 0.20),  # start high and to the side
        (-0.20, 0.0, 0.05),  # descend to push height
        (0.20, 0.0, 0.05),  # sweep through the object along +x
        (0.20, 0.0, 0.20),  # lift back up
    ]
    trajectory = multi_poke_trajectory(waypoints, steps_per_segment=150)

    # Fine interpolation and multiple physics substeps prevent the claw from
    # tunneling through the object during contact.
    frames = sim.rollout(
        interactor_trajectory={"interactor0": trajectory},
        object_bodies=["object"],
        capture_every=8,
        substeps=4,
    )
    if not frames:
        raise RuntimeError("The rollout produced no captured frames")

    pos_start = frames[0].object_qpos["object"][:3]
    pos_end = frames[-1].object_qpos["object"][:3]
    print(f"collected {len(frames)} frames")
    print(f"object moved from {pos_start.round(3)} to {pos_end.round(3)}")
    print(f"displacement along x: {pos_end[0] - pos_start[0]:.3f} m")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_idxs = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    for i in sample_idxs:
        rgb, _ = frames[i].rgbd["cam_side"]
        Image.fromarray(rgb).save(OUTPUT_DIR / f"_frame_{i}.png")

    gif_frames = [Image.fromarray(frame.rgbd["cam_side"][0]) for frame in frames]
    gif_frames[0].save(
        OUTPUT_DIR / "push_demo.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=80,
        loop=0,
    )


if __name__ == "__main__":
    main()
