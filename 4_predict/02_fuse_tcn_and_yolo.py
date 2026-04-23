from __future__ import annotations

"""TCN 프레임 라벨과 YOLO 상태 정보를 결합해 최종 이벤트를 만든다."""

from pathlib import Path

import pandas as pd

from predict_paths import (
    DEFAULT_TCN_PRED_ROOTS,
    EVENT_OUTPUT_ROOT,
    FUSED_FRAME_OUTPUT_ROOT,
    YOLO_STATE_OUTPUT_ROOT,
    YOLO_TEST_ROOT,
    ensure_output_dirs,
)
from predict_workflow import (
    build_events_from_frame_labels,
    discover_tcn_yolo_pairs,
    fuse_tcn_with_yolo,
    load_label_frame_csv,
)


def run_fusion(
    tcn_root_paths: list[Path] | None = None,
    yolo_root_paths: list[Path] | None = None,
    fused_output_root: Path = FUSED_FRAME_OUTPUT_ROOT,
    event_output_root: Path = EVENT_OUTPUT_ROOT,
) -> None:
    """짝이 맞는 TCN/YOLO CSV를 찾아 융합 결과와 이벤트 CSV를 저장한다."""
    ensure_output_dirs()
    tcn_root_paths = tcn_root_paths or DEFAULT_TCN_PRED_ROOTS
    yolo_root_paths = yolo_root_paths or [YOLO_STATE_OUTPUT_ROOT, YOLO_TEST_ROOT / "out_yolo"]

    pairs = discover_tcn_yolo_pairs(tcn_root_paths, yolo_root_paths)
    print(f"[INFO] Found {len(pairs)} TCN/YOLO pairs")

    for pair in pairs:
        tcn_df = load_label_frame_csv(pair.tcn_path)
        yolo_df = pd.read_csv(pair.yolo_path)

        fused_df = fuse_tcn_with_yolo(tcn_df, yolo_df)
        fused_path = fused_output_root / f"yolo_tcn_{pair.sample_id}.csv"
        fused_df.to_csv(fused_path, index=False, encoding="utf-8-sig")

        event_df = build_events_from_frame_labels(fused_df["fused_label"].tolist())
        event_path = event_output_root / f"{pair.sample_id}_events_pred.csv"
        event_df.to_csv(event_path, index=False, encoding="utf-8-sig")

        print(f"[SAVE] {fused_path}")
        print(f"[SAVE] {event_path}")


if __name__ == "__main__":
    run_fusion()
