"""Google Colab 환경에서 프로젝트 경로를 해석하기 위한 유틸리티."""

from pathlib import Path


DEFAULT_PROJECT_DIR = Path("/content/drive/MyDrive/sessac_project")
DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/sessac_project_artifacts")


def resolve_project_paths(project_dir: str | Path) -> dict[str, Path]:
    """Colab에서 쓰던 주요 데이터 경로를 한 번에 계산한다."""
    root = Path(project_dir).expanduser().resolve()
    return {
        "project_root": root,
        "behavior_labels": root / "data" / "labels",
        "behavior_landmarks": root / "data" / "landmarks",
        "behavior_tensors": root / "data" / "tensors",
        "legacy_labels": root / "data" / "out_csv",
        "legacy_landmarks": root / "data" / "out_npz",
        "yolo_images": root / "yolo" / "imege",
        "yolo_labels_open_close": root / "yolo" / "labels_openclose",
        "yolo_labels_full_empty": root / "yolo" / "labels_fullempty",
    }
