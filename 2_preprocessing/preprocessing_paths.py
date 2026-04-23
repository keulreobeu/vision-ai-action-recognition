"""전처리 단계에서 공통으로 사용하는 입력/출력 경로 모음."""

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
    """시나리오별 원본 프레임 폴더를 반환한다."""
    return FRAME_ROOTS[scenario]


def get_label_root(scenario: str) -> Path:
    """시나리오별 프레임 라벨 CSV 저장 폴더를 반환한다."""
    return LABEL_ROOT / scenario


def get_landmark_root(scenario: str) -> Path:
    """시나리오별 손 랜드마크 NPZ 저장 폴더를 반환한다."""
    return LANDMARK_ROOT / scenario


def get_tensor_root(scenario: str) -> Path:
    """시나리오별 윈도우 텐서 저장 폴더를 반환한다."""
    return TENSOR_ROOT / scenario


def find_sample_dirs(scenarios: list[str] | None = None) -> list[tuple[str, Path]]:
    """프레임 이미지가 실제로 들어 있는 샘플 폴더만 수집한다."""
    selected = scenarios or list(FRAME_ROOTS.keys())
    sample_dirs: list[tuple[str, Path]] = []

    for scenario in selected:
        frame_root = get_frame_root(scenario)
        if not frame_root.exists():
            continue

        for candidate in sorted(frame_root.iterdir()):
            if not candidate.is_dir():
                continue

            # 이미지가 하나라도 있으면 유효한 샘플로 간주한다.
            has_frames = any(
                child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png"}
                for child in candidate.iterdir()
            )
            if has_frames:
                sample_dirs.append((scenario, candidate))

    return sample_dirs


def find_event_csv(sample_name: str, scenario: str) -> Path | None:
    """샘플 이름과 시나리오에 맞는 이벤트 CSV를 우선순위대로 찾는다."""
    for root in EVENT_ROOTS.get(scenario, []):
        event_path = root / f"{sample_name}_events.csv"
        if event_path.exists():
            return event_path
    return None
