from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_ROOT = PROJECT_ROOT / "4_predict"

YOLO_ROOT = PROJECT_ROOT / "yolo"
YOLO_TEST_ROOT = YOLO_ROOT / "test_video"
YOLO_OPENCLOSE_MODEL = YOLO_ROOT / "best_openclose.pt"
YOLO_FULLEMPTY_MODEL = YOLO_ROOT / "best_fullempty.pt"

OUTPUT_ROOT = PREDICT_ROOT / "output"
YOLO_STATE_OUTPUT_ROOT = OUTPUT_ROOT / "yolo_states"
FUSED_FRAME_OUTPUT_ROOT = OUTPUT_ROOT / "fused_frames"
EVENT_OUTPUT_ROOT = OUTPUT_ROOT / "events"
METRIC_OUTPUT_ROOT = OUTPUT_ROOT / "metrics"

DEFAULT_TCN_PRED_ROOTS = [
    OUTPUT_ROOT / "tcn_predictions",
    YOLO_TEST_ROOT / "out_TCN",
]

DEFAULT_GT_LABEL_ROOTS = [
    PROJECT_ROOT / "data" / "labels",
    PROJECT_ROOT / "Backup" / "old" / "test_data" / "test_in_model" / "test_csv",
]

DEFAULT_GT_EVENT_ROOTS = [
    PREDICT_ROOT / "input" / "gt_events",
    PROJECT_ROOT / "Backup" / "old" / "test_data" / "test_flagle",
]


def ensure_output_dirs() -> None:
    for path in (
        OUTPUT_ROOT,
        YOLO_STATE_OUTPUT_ROOT,
        FUSED_FRAME_OUTPUT_ROOT,
        EVENT_OUTPUT_ROOT,
        METRIC_OUTPUT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
