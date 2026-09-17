"""Shared controller-label handling for human and robotic interactors."""

DEFAULT_CONTROLLER_NAMES = (
    "hand",
    "claw",
    "gripper",
    "robot gripper",
    "robotic gripper",
    "robot",
)


def parse_controller_names(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_CONTROLLER_NAMES)
    return {name.strip().lower() for name in value.split(",") if name.strip()}


def is_controller_label(label: str, names: set[str]) -> bool:
    normalized = label.strip().lower().rstrip(".")
    return normalized in names
