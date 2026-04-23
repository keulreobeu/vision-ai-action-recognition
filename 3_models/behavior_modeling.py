"""행동 인식용 데이터 적재, 폴드 분할, 모델 정의를 모아 둔 모듈."""

import copy
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LABEL_COLUMNS = ("A", "S", "D")


@dataclass
class SampleRecord:
    """샘플 하나의 라벨/랜드마크 파일과 메타데이터를 담는다."""
    sample_name: str
    scenario: str
    label_path: Path
    landmark_path: Path
    set_id: str
    sequence_length: int


def _extract_metadata(scenario: str, sample_stem: str) -> tuple[str, int]:
    """파일 이름 규칙에서 세트 구성을 위한 접두사와 번호를 추출한다."""
    if scenario == "normal":
        match = re.match(r"video_normal_(\d+)$", sample_stem)
        if not match:
            raise ValueError(f"Unexpected normal sample name: {sample_stem}")
        sample_number = int(match.group(1))
        return "normal", sample_number

    if scenario.startswith("missing"):
        match = re.match(rf"video_{re.escape(scenario)}_([A-Z])_(\d+)$", sample_stem)
        if not match:
            raise ValueError(f"Unexpected missing sample name: {sample_stem}")
        variant = match.group(1)
        sample_number = int(match.group(2))
        return f"{scenario}_{variant}", sample_number

    if scenario == "idle":
        match = re.match(r"video_idle_(\d+)$", sample_stem)
        if not match:
            raise ValueError(f"Unexpected idle sample name: {sample_stem}")
        sample_number = int(match.group(1))
        return "idle", sample_number

    raise ValueError(f"Unknown scenario: {scenario}")


def _load_label_array(label_path: Path) -> np.ndarray:
    """CSV에서 A/S/D 라벨만 골라 float 배열로 읽는다."""
    frame = pd.read_csv(label_path)
    available_columns = {column.lower(): column for column in frame.columns}
    selected_columns = []
    for label in LABEL_COLUMNS:
        column_name = available_columns.get(label.lower())
        if column_name is None:
            raise ValueError(f"Missing label column '{label}' in {label_path}")
        selected_columns.append(column_name)
    return frame[selected_columns].to_numpy(dtype=np.float32)


def discover_behavior_samples(
    labels_root: Path,
    landmarks_root: Path,
    legacy_labels_root: Path | None = None,
    legacy_landmarks_root: Path | None = None,
    scenarios: list[str] | None = None,
) -> list[SampleRecord]:
    """학습 가능한 라벨/랜드마크 샘플 조합을 찾아 메타데이터를 만든다."""
    if scenarios is not None:
        selected_scenarios = scenarios
    else:
        selected_scenarios = []
        for root in (labels_root, legacy_labels_root):
            if root is None or not root.exists():
                continue
            for path in root.iterdir():
                if path.is_dir() and path.name not in selected_scenarios:
                    selected_scenarios.append(path.name)
        selected_scenarios = sorted(selected_scenarios)

    records: list[SampleRecord] = []

    for scenario in selected_scenarios:
        label_dir = labels_root / scenario
        landmark_dir = landmarks_root / scenario
        label_suffix = "_labels.csv"

        # 정리된 폴더가 없으면 기존 레거시 산출물도 후순위로 탐색한다.
        if not label_dir.exists() and legacy_labels_root is not None:
            label_dir = legacy_labels_root / scenario
            landmark_dir = (legacy_landmarks_root or landmarks_root) / scenario
            label_suffix = "_lange.csv"

        if not label_dir.exists() or not landmark_dir.exists():
            continue

        for label_path in sorted(label_dir.glob(f"*{label_suffix}")):
            sample_stem = label_path.stem.replace(label_suffix.replace(".csv", ""), "")
            landmark_path = landmark_dir / f"hands_{sample_stem}.npz"
            if not landmark_path.exists():
                continue

            labels = _load_label_array(label_path)
            landmark_data = np.load(landmark_path)
            if "hand_kps" not in landmark_data:
                continue

            landmarks = landmark_data["hand_kps"].astype(np.float32)
            sequence_length = min(len(labels), len(landmarks))
            if sequence_length == 0:
                continue

            type_prefix, sample_number = _extract_metadata(scenario, sample_stem)
            set_index = (sample_number - 1) // 10 + 1
            records.append(
                SampleRecord(
                    sample_name=f"{scenario}/{sample_stem}",
                    scenario=scenario,
                    label_path=label_path,
                    landmark_path=landmark_path,
                    set_id=f"{type_prefix}_set{set_index}",
                    sequence_length=sequence_length,
                )
            )

    return records


def load_behavior_arrays(records: list[SampleRecord]) -> dict[str, dict[str, np.ndarray]]:
    """샘플 메타데이터 목록을 실제 학습용 numpy 배열로 적재한다."""
    data: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        labels = _load_label_array(record.label_path)[: record.sequence_length]
        landmarks = np.load(record.landmark_path)["hand_kps"].astype(np.float32)[: record.sequence_length]
        data[record.sample_name] = {
            "labels": labels,
            "landmarks": landmarks,
            "scenario": record.scenario,
            "set_id": record.set_id,
        }
    return data


def build_group_folds(records: list[SampleRecord], num_folds: int = 4) -> list[dict[str, list[str]]]:
    """같은 세트가 train/val에 동시에 섞이지 않도록 그룹 기반 폴드를 만든다."""
    grouped: dict[str, list[str]] = {}
    for record in records:
        grouped.setdefault(record.set_id, []).append(record.sample_name)

    set_ids = sorted(grouped)
    folds: list[list[str]] = [[] for _ in range(num_folds)]
    for index, set_id in enumerate(set_ids):
        folds[index % num_folds].append(set_id)

    split_info = []
    for fold_index in range(num_folds):
        val_set_ids = set(folds[fold_index])
        train_keys: list[str] = []
        val_keys: list[str] = []
        for set_id, sample_names in grouped.items():
            if set_id in val_set_ids:
                val_keys.extend(sample_names)
            else:
                train_keys.extend(sample_names)
        split_info.append(
            {
                "fold_index": fold_index,
                "train_keys": sorted(train_keys),
                "val_keys": sorted(val_keys),
                "val_set_ids": sorted(val_set_ids),
            }
        )
    return split_info


class LandmarkWindowDataset(Dataset):
    """랜드마크 시퀀스를 슬라이딩 윈도우 단위로 잘라 제공하는 Dataset."""
    def __init__(
        self,
        data_dict: dict[str, dict[str, np.ndarray]],
        sample_names: list[str],
        window_size: int = 15,
        step_size: int = 5,
    ) -> None:
        self.data_dict = data_dict
        self.sample_names = sample_names
        self.window_size = window_size
        self.step_size = step_size
        self.items: list[tuple[str, int, int]] = []

        for sample_name in sample_names:
            sequence = data_dict[sample_name]["landmarks"]
            for start in range(0, len(sequence) - window_size + 1, step_size):
                self.items.append((sample_name, start, start + window_size))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample_name, start, end = self.items[index]
        sample = self.data_dict[sample_name]
        x_window = torch.from_numpy(sample["landmarks"][start:end]).float()
        y_window = torch.from_numpy(sample["labels"][start:end]).float()
        return {
            "x": x_window,
            "y_seq": y_window,
            "y_last": y_window[-1],
            "sample_name": sample_name,
            "start": start,
            "end": end,
        }


class MLPTemporalPoolingClassifier(nn.Module):
    """프레임별 특징을 MLP로 변환한 뒤 시간축 평균으로 분류하는 모델."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.feature(x).mean(dim=1)
        return self.head(pooled)


class MLPAvgPoolClassifier(nn.Module):
    """입력 시퀀스를 평균 풀링한 뒤 얕은 MLP로 분류하는 기준 모델."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.mlp(pooled)


class Chomp1d(nn.Module):
    """TCN의 causal padding 뒤쪽을 잘라 길이를 맞추는 보조 모듈."""
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN의 기본 residual block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(self.net(x) + residual)


class TCNClassifier(nn.Module):
    """시간축 인과 합성곱으로 마지막 시점 라벨을 예측하는 모델."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        channels: tuple[int, ...] = (64, 64),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = input_dim
        for block_index, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.network(x)
        last_step = features[:, :, -1]
        return self.head(last_step)


class SimpleCNN1DClassifier(nn.Module):
    """평균 풀링 기반 1D CNN 분류기."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        channels: tuple[int, ...] = (64, 128),
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_dim
        padding = kernel_size // 2
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.network(x)
        pooled = features.mean(dim=2)
        return self.head(pooled)


class BiLSTMClassifier(nn.Module):
    """양방향 LSTM으로 시퀀스 전체 문맥을 반영하는 분류기."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = self.dropout(output[:, -1, :])
        return self.head(last_step)


def build_dataloaders(
    data_dict: dict[str, dict[str, np.ndarray]],
    fold_info: dict[str, list[str]],
    batch_size: int = 64,
    window_size: int = 15,
    step_size: int = 5,
) -> tuple[LandmarkWindowDataset, LandmarkWindowDataset, DataLoader, DataLoader]:
    """폴드 정보에 맞는 train/val 데이터셋과 DataLoader를 만든다."""
    train_dataset = LandmarkWindowDataset(data_dict, fold_info["train_keys"], window_size, step_size)
    val_dataset = LandmarkWindowDataset(data_dict, fold_info["val_keys"], window_size, step_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """학습 데이터로 한 epoch를 수행하고 손실/정확도를 반환한다."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y_last"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_examples += x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).all(dim=1).sum().item()

    return total_loss / total_examples, total_correct / total_examples


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """검증 데이터로 한 epoch를 평가한다."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y_last"].to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_examples += x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).all(dim=1).sum().item()

    return total_loss / total_examples, total_correct / total_examples


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    patience: int = 10,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """조기 종료를 포함한 전체 학습 루프를 실행한다."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, history


def summarize_history(history: list[dict[str, float]]) -> dict[str, float]:
    """학습 이력에서 가장 좋은 검증 손실 기준 결과를 요약한다."""
    best_row = min(history, key=lambda row: row["val_loss"])
    return {
        "best_epoch": best_row["epoch"],
        "best_val_loss": best_row["val_loss"],
        "best_val_acc": best_row["val_acc"],
        "best_train_loss": best_row["train_loss"],
        "best_train_acc": best_row["train_acc"],
    }
