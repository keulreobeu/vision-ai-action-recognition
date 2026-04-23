"""YOLO 데이터셋 분할과 학습 실행을 위한 보조 함수 모음."""

import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


IMG_SUFFIXES = (".jpg", ".jpeg", ".png")


def discover_yolo_bases(image_root: Path, label_root: Path) -> list[str]:
    """이미지와 라벨이 모두 존재하는 샘플 기준 이름만 추린다."""
    valid_bases = []
    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMG_SUFFIXES:
            continue
        base = image_path.stem
        if (label_root / f"{base}.txt").exists():
            valid_bases.append(base)
    return valid_bases


def prepare_yolo_dataset(
    image_root: Path,
    label_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Path | int]:
    """원본 이미지/라벨을 train/val 구조로 복사해 YOLO 학습셋을 만든다."""
    random.seed(seed)
    valid_bases = discover_yolo_bases(image_root, label_root)
    random.shuffle(valid_bases)

    val_count = int(len(valid_bases) * val_ratio)
    val_bases = set(valid_bases[:val_count])
    train_bases = set(valid_bases[val_count:])

    # Ultralytics 기본 폴더 구조를 미리 만들어 둔다.
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for base in valid_bases:
        split = "train" if base in train_bases else "val"
        src_image = next((image_root / f"{base}{suffix}" for suffix in IMG_SUFFIXES if (image_root / f"{base}{suffix}").exists()), None)
        if src_image is None:
            continue
        shutil.copy2(src_image, output_root / "images" / split / src_image.name)
        shutil.copy2(label_root / f"{base}.txt", output_root / "labels" / split / f"{base}.txt")

    return {
        "dataset_root": output_root,
        "train_count": len(train_bases),
        "val_count": len(val_bases),
    }


def write_yolo_yaml(dataset_root: Path, yaml_path: Path, class_names: list[str]) -> Path:
    """Ultralytics가 읽을 수 있는 dataset YAML 파일을 생성한다."""
    content = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path


def train_yolo_model(
    yaml_path: Path,
    model_name: str,
    project_dir: Path,
    run_name: str,
    epochs: int = 50,
    image_size: int = 640,
    batch_size: int = 16,
    device: int | str = 0,
) -> dict[str, str | float]:
    """지정한 설정으로 YOLO 학습을 실행하고 핵심 결과만 반환한다."""
    model = YOLO(model_name)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=str(project_dir),
        name=run_name,
        device=device,
        exist_ok=True,
    )
    metrics = getattr(results, "results_dict", {})
    return {
        "run_dir": str(project_dir / run_name),
        "best_weights": str(project_dir / run_name / "weights" / "best.pt"),
        "map50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
    }
