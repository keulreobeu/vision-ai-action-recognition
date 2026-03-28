from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_ROOT = Path(__file__).resolve().parent

FRAME_ROOTS = {
    "normal": PROJECT_ROOT / "video" / "video",
    "missing1": PROJECT_ROOT / "video" / "missing1",
    "missing2": PROJECT_ROOT / "video" / "missing2",
    "idle": PROJECT_ROOT / "video" / "idle",
}

EVENT_ROOTS = {
    "normal": [
        PROJECT_ROOT / "data" / "out_csv" / "normal",
        PROJECT_ROOT / "video" / "video" / "normal",
    ],
    "missing1": [
        PROJECT_ROOT / "data" / "out_csv" / "missing1",
    ],
    "missing2": [
        PROJECT_ROOT / "data" / "out_csv" / "missing2",
    ],
    "idle": [],
}

LABEL_ROOT = PROJECT_ROOT / "data" / "labels"
LANDMARK_ROOT = PROJECT_ROOT / "data" / "landmarks"
TENSOR_ROOT = PROJECT_ROOT / "data" / "tensors"

YOLO_XML_ROOT = PROJECT_ROOT / "yolo" / "label"
YOLO_OPEN_CLOSE_ROOT = PROJECT_ROOT / "yolo" / "labels_open_close"
YOLO_FULL_EMPTY_ROOT = PROJECT_ROOT / "yolo" / "labels_full_empty"


def get_frame_root(scenario: str) -> Path:
    return FRAME_ROOTS[scenario]


def get_label_root(scenario: str) -> Path:
    return LABEL_ROOT / scenario


def get_landmark_root(scenario: str) -> Path:
    return LANDMARK_ROOT / scenario


def get_tensor_root(scenario: str) -> Path:
    return TENSOR_ROOT / scenario


def find_sample_dirs(scenarios: list[str] | None = None) -> list[tuple[str, Path]]:
    selected = scenarios or list(FRAME_ROOTS.keys())
    sample_dirs: list[tuple[str, Path]] = []

    for scenario in selected:
        frame_root = get_frame_root(scenario)
        if not frame_root.exists():
            continue

        for candidate in sorted(frame_root.iterdir()):
            if not candidate.is_dir():
                continue

            has_frames = any(
                child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png"}
                for child in candidate.iterdir()
            )
            if has_frames:
                sample_dirs.append((scenario, candidate))

    return sample_dirs


def find_event_csv(sample_name: str, scenario: str) -> Path | None:
    for root in EVENT_ROOTS.get(scenario, []):
        event_path = root / f"{sample_name}_events.csv"
        if event_path.exists():
            return event_path
    return None
