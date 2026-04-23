from __future__ import annotations

"""TCN 예측과 YOLO 박스 상태를 정렬하고 평가하는 공통 로직 모음."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_COLUMNS = ["A", "S", "D"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SamplePair:
    """같은 샘플에 대응하는 TCN/YOLO CSV 경로 쌍."""
    sample_id: str
    tcn_path: Path
    yolo_path: Path


def normalize_sample_id(raw_name: str) -> str:
    """여러 단계에서 달라진 파일 접미사를 제거해 공통 샘플 ID로 맞춘다."""
    text = Path(raw_name).stem
    for suffix in (
        "_yolo_states",
        "_events_pred",
        "_events",
        "_pred",
        "_labels",
        "_label",
        "_flag",
        "_flage",
        "_lange",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace(" ", "")
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def list_frame_dirs(root: Path) -> list[Path]:
    """예측 대상 프레임 폴더만 골라 반환한다."""
    if not root.exists():
        return []
    ignored = {"out_yolo", "out_TCN", "out_pred"}
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name not in ignored
    )


def list_frame_images(frames_dir: Path) -> list[Path]:
    """프레임 폴더 안의 이미지 파일을 정렬해서 반환한다."""
    return sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def discover_csvs(root_paths: Iterable[Path], suffix: str) -> dict[str, Path]:
    """여러 루트에서 접미사에 맞는 CSV를 찾아 샘플 ID 기준으로 묶는다."""
    discovered: dict[str, Path] = {}
    for root_path in root_paths:
        if not root_path.exists():
            continue
        for csv_path in root_path.rglob(f"*{suffix}"):
            sample_id = normalize_sample_id(csv_path.name)
            discovered.setdefault(sample_id, csv_path)
    return discovered


def discover_tcn_yolo_pairs(
    tcn_root_paths: Iterable[Path],
    yolo_root_paths: Iterable[Path],
) -> list[SamplePair]:
    """같은 샘플을 가리키는 TCN 예측 CSV와 YOLO 상태 CSV를 짝지어 찾는다."""
    tcn_csvs = discover_csvs(tcn_root_paths, ".csv")
    yolo_csvs = discover_csvs(yolo_root_paths, ".csv")
    sample_ids = sorted(set(tcn_csvs) & set(yolo_csvs))
    pairs: list[SamplePair] = []
    for sample_id in sample_ids:
        tcn_path = tcn_csvs[sample_id]
        yolo_path = yolo_csvs[sample_id]
        if "_yolo_states" not in yolo_path.stem:
            continue
        if not (
            tcn_path.stem.endswith("_pred")
            or tcn_path.stem.endswith("_labels")
            or tcn_path.stem.endswith("_lange")
        ):
            continue
        pairs.append(SamplePair(sample_id=sample_id, tcn_path=tcn_path, yolo_path=yolo_path))
    return pairs


def load_label_frame_csv(csv_path: Path) -> pd.DataFrame:
    """A/S/D 라벨 컬럼만 남긴 표준 프레임 라벨 DataFrame을 만든다."""
    df = pd.read_csv(csv_path)
    missing = [column for column in LABEL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing label columns: {missing}")
    label_df = df[LABEL_COLUMNS].copy()
    label_df = label_df.fillna(0).astype(int)
    return label_df


def derive_tcn_label(row: pd.Series) -> str:
    """멀티라벨 행을 단일 대표 라벨로 축약한다."""
    labels = [label for label in LABEL_COLUMNS if int(row.get(label, 0)) == 1]
    if len(labels) == 1:
        return labels[0]
    return "idle"


def build_events_from_frame_labels(labels: list[str]) -> pd.DataFrame:
    """프레임 라벨 시퀀스를 이벤트 시작 프레임 목록으로 변환한다."""
    rows: list[dict[str, object]] = []
    previous = "idle"
    flag_id = 0
    for frame_idx, label in enumerate(labels):
        if label == previous:
            continue
        if label != "idle":
            flag_id += 1
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "flag_id": flag_id,
                    "flag_key": label,
                }
            )
        previous = label
    return pd.DataFrame(rows, columns=["frame_idx", "flag_id", "flag_key"])


def fuse_tcn_with_yolo(tcn_df: pd.DataFrame, yolo_df: pd.DataFrame) -> pd.DataFrame:
    """TCN 라벨과 YOLO 상태 특징을 결합해 최종 프레임 라벨을 계산한다."""
    n_rows = min(len(tcn_df), len(yolo_df))
    fused = pd.concat(
        [
            tcn_df.iloc[:n_rows].reset_index(drop=True).copy(),
            yolo_df.iloc[:n_rows].reset_index(drop=True).copy(),
        ],
        axis=1,
    )
    fused["tcn_label"] = fused.apply(derive_tcn_label, axis=1)

    empty_count = fused.get("empty_count", pd.Series([0] * n_rows))
    full_count = fused.get("full_count", pd.Series([0] * n_rows))
    open_count = fused.get("open_count", pd.Series([0] * n_rows))
    closed_count = fused.get("closed_count", pd.Series([0] * n_rows))

    # 박스 개수의 증감을 이용해 상태 전환 신호를 추정한다.
    diff_empty = empty_count.diff().fillna(0)
    diff_full = full_count.diff().fillna(0)

    fused["yolo_A_like"] = (open_count > 0) | ((diff_empty < 0) & (diff_full >= 0))
    fused["yolo_S_like"] = (closed_count > 0) & (empty_count > 0)
    fused["yolo_D_like"] = (full_count > 0) | ((diff_full > 0) & (diff_empty <= 0))

    fused["fused_label_raw"] = fused["tcn_label"]
    # TCN을 주 신호로 두고 YOLO를 보조 신호로 더해 최종 라벨을 고른다.
    for index, row in fused.iterrows():
        scores = {
            "A": 2.0 * float(row["tcn_label"] == "A") + 1.0 * float(row["yolo_A_like"]),
            "S": 2.0 * float(row["tcn_label"] == "S") + 1.0 * float(row["yolo_S_like"]),
            "D": 2.0 * float(row["tcn_label"] == "D") + 1.0 * float(row["yolo_D_like"]),
            "idle": 2.0 * float(row["tcn_label"] == "idle"),
        }
        fused.at[index, "fused_label_raw"] = max(scores, key=scores.get)

    # 너무 짧은 구간은 노이즈로 보고 idle로 평활화한다.
    smoothed_labels = fused["fused_label_raw"].tolist()
    segment_start = 0
    for index in range(1, len(smoothed_labels) + 1):
        segment_ended = index == len(smoothed_labels) or smoothed_labels[index] != smoothed_labels[segment_start]
        if not segment_ended:
            continue
        if index - segment_start < 5:
            for replace_index in range(segment_start, index):
                smoothed_labels[replace_index] = "idle"
        segment_start = index
    fused["fused_label"] = smoothed_labels
    return fused


def exact_frame_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """프레임 단위 완전 일치 정확도를 계산한다."""
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).all(axis=1).mean())


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """클래스별 및 마이크로 평균 다중라벨 지표를 계산한다."""
    eps = 1e-12
    per_class: dict[str, dict[str, float]] = {}
    total_tp = total_fp = total_fn = total_tn = 0
    for index, label in enumerate(LABEL_COLUMNS):
        true_column = y_true[:, index].astype(bool)
        pred_column = y_pred[:, index].astype(bool)
        tp = int(np.logical_and(true_column, pred_column).sum())
        fp = int(np.logical_and(~true_column, pred_column).sum())
        fn = int(np.logical_and(true_column, ~pred_column).sum())
        tn = int(np.logical_and(~true_column, ~pred_column).sum())
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": acc,
            "support": int(true_column.sum()),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    micro_precision = total_tp / (total_tp + total_fp + eps)
    micro_recall = total_tp / (total_tp + total_fn + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    micro_acc = (total_tp + total_tn) / max(total_tp + total_fp + total_fn + total_tn, 1)
    overall = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_acc": micro_acc,
    }
    return overall, per_class
