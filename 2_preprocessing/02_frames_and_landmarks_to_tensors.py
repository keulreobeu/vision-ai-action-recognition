"""프레임, 손 랜드마크, 라벨을 윈도우 단위 텐서 묶음으로 저장한다."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from preprocessing_paths import (
    find_sample_dirs,
    get_label_root,
    get_landmark_root,
    get_tensor_root,
)


FRAME_INDEX_COLUMN = "frame_idx"
LABEL_COLUMNS = ("A", "S", "D")


def parse_args() -> argparse.Namespace:
    """윈도우 크기 등 텐서화 옵션을 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Convert frames, landmark NPZ files, and per-frame labels into window tensors.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["normal", "missing1", "missing2", "idle"],
        choices=["normal", "missing1", "missing2", "idle"],
        help="Scenario folders to process.",
    )
    parser.add_argument("--window", type=int, default=15, help="Window size in frames.")
    parser.add_argument("--step", type=int, default=5, help="Step size between windows.")
    parser.add_argument("--image-size", type=int, default=224, help="Resized image size.")
    return parser.parse_args()


def list_frame_files(frames_dir: Path) -> list[Path]:
    """프레임 폴더의 이미지 파일 목록을 정렬해 반환한다."""
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def build_window_indices(frame_count: int, window: int, step: int) -> list[tuple[int, int]]:
    """슬라이딩 윈도우 시작/종료 인덱스를 계산한다."""
    return [
        (start, start + window)
        for start in range(0, frame_count - window + 1, step)
    ]


def load_label_matrix(label_csv_path: Path, frame_count: int) -> np.ndarray:
    """CSV 라벨을 고정 길이 프레임 라벨 행렬로 읽어 온다."""
    label_frame = pd.read_csv(label_csv_path)
    labels = np.zeros((frame_count, len(LABEL_COLUMNS)), dtype=np.float32)

    if FRAME_INDEX_COLUMN in label_frame.columns:
        for _, row in label_frame.sort_values(FRAME_INDEX_COLUMN).iterrows():
            frame_index = int(row[FRAME_INDEX_COLUMN])
            if 0 <= frame_index < frame_count:
                labels[frame_index] = row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    else:
        raw = label_frame[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        valid_count = min(frame_count, len(raw))
        labels[:valid_count] = raw[:valid_count]

    return labels


def load_frame_tensor(frame_paths: list[Path], transform: transforms.Compose) -> torch.Tensor:
    """이미지 프레임 리스트를 모델 입력용 텐서로 쌓는다."""
    images = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        images.append(transform(image))
    return torch.stack(images, dim=0)


def process_sample(
    scenario: str,
    sample_dir: Path,
    window: int,
    step: int,
    image_size: int,
) -> None:
    """샘플 하나를 윈도우 텐서 파일로 변환한다."""
    sample_name = sample_dir.name
    frame_paths = list_frame_files(sample_dir)
    if not frame_paths:
        print(f"[SKIP] {sample_name}: no frame images found")
        return

    label_csv_path = get_label_root(scenario) / f"{sample_name}_labels.csv"
    landmark_path = get_landmark_root(scenario) / f"hands_{sample_name}.npz"
    if not label_csv_path.exists():
        print(f"[SKIP] {sample_name}: missing label CSV -> {label_csv_path}")
        return
    if not landmark_path.exists():
        print(f"[SKIP] {sample_name}: missing landmark NPZ -> {landmark_path}")
        return

    landmark_data = np.load(landmark_path)
    hand_keypoints = landmark_data["hand_kps"]
    # 프레임 수와 랜드마크 길이가 다르면 공통 구간만 사용한다.
    frame_count = min(len(frame_paths), hand_keypoints.shape[0])
    frame_paths = frame_paths[:frame_count]
    hand_keypoints = hand_keypoints[:frame_count]

    label_matrix = load_label_matrix(label_csv_path, frame_count)
    window_indices = build_window_indices(frame_count, window=window, step=step)
    if not window_indices:
        print(f"[SKIP] {sample_name}: frame count {frame_count} is smaller than window {window}")
        return

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    image_windows: list[torch.Tensor] = []
    hand_windows: list[torch.Tensor] = []
    label_sequence_windows: list[torch.Tensor] = []
    label_last_windows: list[torch.Tensor] = []

    for start, end in window_indices:
        frame_window = load_frame_tensor(frame_paths[start:end], transform)
        hand_window = torch.from_numpy(hand_keypoints[start:end]).float()
        label_window = torch.from_numpy(label_matrix[start:end]).float()

        image_windows.append(frame_window)
        hand_windows.append(hand_window)
        label_sequence_windows.append(label_window)
        label_last_windows.append(label_window[-1])

    output_dir = get_tensor_root(scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_name}_windows.pt"

    save_payload = {
        "X_img": torch.stack(image_windows, dim=0),
        "X_hand": torch.stack(hand_windows, dim=0),
        "y_seq": torch.stack(label_sequence_windows, dim=0),
        "y_last": torch.stack(label_last_windows, dim=0),
        "sample_name": sample_name,
        "window": window,
        "step": step,
        "image_size": image_size,
        "num_actions": len(LABEL_COLUMNS),
        "hand_dim": hand_keypoints.shape[1],
    }
    torch.save(save_payload, output_path)
    print(f"[TENSOR] {sample_name}: {output_path}")


def main() -> None:
    """요청한 시나리오 전체를 텐서화한다."""
    args = parse_args()
    sample_dirs = find_sample_dirs(args.scenarios)

    if not sample_dirs:
        print("[ERROR] No frame directories found for the requested scenarios.")
        return

    for scenario, sample_dir in sample_dirs:
        process_sample(
            scenario=scenario,
            sample_dir=sample_dir,
            window=args.window,
            step=args.step,
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
