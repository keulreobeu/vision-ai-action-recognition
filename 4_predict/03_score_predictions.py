from __future__ import annotations

"""GT 라벨과 예측 결과를 비교해 프레임 단위 성능 지표를 계산한다."""

from pathlib import Path

import numpy as np
import pandas as pd

from predict_paths import DEFAULT_GT_LABEL_ROOTS, FUSED_FRAME_OUTPUT_ROOT, METRIC_OUTPUT_ROOT, ensure_output_dirs
from predict_workflow import (
    LABEL_COLUMNS,
    compute_multilabel_metrics,
    discover_csvs,
    exact_frame_accuracy,
    load_label_frame_csv,
)


def load_prediction_frame_csv(csv_path: Path) -> pd.DataFrame:
    """예측 CSV를 표준 A/S/D 다중라벨 형태로 정규화한다."""
    df = pd.read_csv(csv_path)
    if all(column in df.columns for column in LABEL_COLUMNS):
        label_df = df[LABEL_COLUMNS].copy()
    elif "fused_label" in df.columns:
        label_df = pd.DataFrame(0, index=df.index, columns=LABEL_COLUMNS)
        for label in LABEL_COLUMNS:
            label_df[label] = (df["fused_label"] == label).astype(int)
    else:
        raise ValueError(f"{csv_path} does not contain prediction labels")
    return label_df.fillna(0).astype(int)


def evaluate_predictions(
    gt_root_paths: list[Path] | None = None,
    pred_root: Path = FUSED_FRAME_OUTPUT_ROOT,
    metric_output_root: Path = METRIC_OUTPUT_ROOT,
) -> pd.DataFrame:
    """매칭되는 GT/예측 파일 쌍을 평가하고 요약 CSV를 저장한다."""
    ensure_output_dirs()
    gt_root_paths = gt_root_paths or DEFAULT_GT_LABEL_ROOTS

    gt_csvs = discover_csvs(gt_root_paths, ".csv")
    pred_csvs = discover_csvs([pred_root], ".csv")

    rows: list[dict[str, object]] = []
    matched_ids = sorted(set(gt_csvs) & set(pred_csvs))
    print(f"[INFO] Matched {len(matched_ids)} GT/prediction pairs")

    for sample_id in matched_ids:
        gt_df = load_label_frame_csv(gt_csvs[sample_id])
        pred_df = load_prediction_frame_csv(pred_csvs[sample_id])
        n_rows = min(len(gt_df), len(pred_df))
        if n_rows == 0:
            continue

        # 길이가 다를 수 있으므로 공통 길이까지만 비교한다.
        y_true = gt_df.iloc[:n_rows].to_numpy(dtype=int)
        y_pred = pred_df.iloc[:n_rows].to_numpy(dtype=int)
        overall, per_class = compute_multilabel_metrics(y_true, y_pred)

        row = {
            "sample_id": sample_id,
            "n_frames_eval": n_rows,
            "exact_frame_acc": exact_frame_accuracy(y_true, y_pred),
            **overall,
            "gt_path": str(gt_csvs[sample_id]),
            "pred_path": str(pred_csvs[sample_id]),
        }
        for label in LABEL_COLUMNS:
            for metric_name, metric_value in per_class[label].items():
                row[f"{label}_{metric_name}"] = metric_value
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("sample_id").reset_index(drop=True)
    summary_path = metric_output_root / "frame_metrics.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        numeric_columns = summary_df.select_dtypes(include=[np.number]).columns
        average_row = {"sample_id": "__mean__"}
        for column in numeric_columns:
            average_row[column] = float(summary_df[column].mean())
        mean_df = pd.DataFrame([average_row])
        mean_path = metric_output_root / "frame_metrics_mean.csv"
        mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {mean_path}")

    print(f"[SAVE] {summary_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_predictions()
