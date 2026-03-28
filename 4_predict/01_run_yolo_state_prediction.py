from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from predict_paths import (
    YOLO_FULLEMPTY_MODEL,
    YOLO_OPENCLOSE_MODEL,
    YOLO_STATE_OUTPUT_ROOT,
    YOLO_TEST_ROOT,
    ensure_output_dirs,
)
from predict_workflow import list_frame_dirs, list_frame_images, normalize_sample_id


def count_classes(result, names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in names}
    if result.boxes is None or result.boxes.cls is None:
        return counts
    class_ids = result.boxes.cls.tolist()
    for class_id in class_ids:
        label_name = result.names.get(int(class_id), str(int(class_id)))
        if label_name in counts:
            counts[label_name] += 1
    return counts


def run_yolo_state_prediction(
    frames_root: Path = YOLO_TEST_ROOT,
    openclose_model_path: Path = YOLO_OPENCLOSE_MODEL,
    fullempty_model_path: Path = YOLO_FULLEMPTY_MODEL,
    output_root: Path = YOLO_STATE_OUTPUT_ROOT,
) -> None:
    ensure_output_dirs()
    output_root.mkdir(parents=True, exist_ok=True)

    openclose_model = YOLO(str(openclose_model_path))
    fullempty_model = YOLO(str(fullempty_model_path))

    frame_dirs = list_frame_dirs(frames_root)
    print(f"[INFO] Found {len(frame_dirs)} frame folders under {frames_root}")

    for frames_dir in frame_dirs:
        image_paths = list_frame_images(frames_dir)
        if not image_paths:
            print(f"[WARN] No image files found in {frames_dir}")
            continue

        sample_id = normalize_sample_id(frames_dir.name)
        rows: list[dict[str, object]] = []
        for frame_idx, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[WARN] Failed to read {image_path}")
                continue

            oc_result = openclose_model(frame, verbose=False)[0]
            fe_result = fullempty_model(frame, verbose=False)[0]

            openclose_counts = count_classes(oc_result, ["open_box", "closed_box"])
            fullempty_counts = count_classes(fe_result, ["full_box", "empty_box"])
            box_count = (
                openclose_counts["open_box"]
                + openclose_counts["closed_box"]
                + fullempty_counts["full_box"]
                + fullempty_counts["empty_box"]
            )

            rows.append(
                {
                    "video_name": sample_id,
                    "frame_idx": frame_idx,
                    "frame_name": image_path.name,
                    "box_count": box_count,
                    "open_count": openclose_counts["open_box"],
                    "closed_count": openclose_counts["closed_box"],
                    "full_count": fullempty_counts["full_box"],
                    "empty_count": fullempty_counts["empty_box"],
                }
            )

        output_path = output_root / f"{sample_id}_yolo_states.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    run_yolo_state_prediction()
