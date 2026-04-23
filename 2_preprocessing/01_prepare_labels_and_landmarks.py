"""프레임 폴더에서 프레임 라벨 CSV와 손 랜드마크 NPZ를 생성한다."""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from preprocessing_paths import (
    EVENT_ROOTS,
    find_event_csv,
    find_sample_dirs,
    get_label_root,
    get_landmark_root,
)


ACTION_KEYS = ("A", "S", "D")
ACTION_INDEX = {key: index for index, key in enumerate(ACTION_KEYS)}
EVENT_FRAME_COLUMN = "frame_idx"
EVENT_KEY_COLUMN = "flag_key"
MAX_HANDS = 2


def parse_args() -> argparse.Namespace:
    """CLI 실행 옵션을 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Generate per-frame labels and MediaPipe hand landmarks from frame folders.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(EVENT_ROOTS.keys()),
        choices=list(EVENT_ROOTS.keys()),
        help="Scenario folders to process.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=MAX_HANDS,
        help="Maximum number of hands to keep per frame.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip label CSV generation.",
    )
    parser.add_argument(
        "--skip-landmarks",
        action="store_true",
        help="Skip MediaPipe landmark extraction.",
    )
    return parser.parse_args()


def list_frame_files(frames_dir: Path) -> list[Path]:
    """프레임 폴더 안의 이미지 파일만 정렬해서 반환한다."""
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def build_interval_labels(frame_count: int, event_csv_path: Path | None) -> np.ndarray:
    """이벤트 토글 CSV를 프레임 단위 A/S/D 라벨 행렬로 변환한다."""
    labels = np.zeros((frame_count, len(ACTION_KEYS)), dtype=np.float32)
    if event_csv_path is None:
        return labels

    event_frame = pd.read_csv(event_csv_path)
    if event_frame.empty:
        return labels

    event_frame = event_frame.sort_values(EVENT_FRAME_COLUMN)
    state = np.zeros(len(ACTION_KEYS), dtype=np.float32)
    last_frame_index = 0

    for _, row in event_frame.iterrows():
        frame_index = int(row[EVENT_FRAME_COLUMN])
        key = str(row[EVENT_KEY_COLUMN]).strip().upper()
        if key not in ACTION_INDEX:
            continue

        frame_index = max(0, min(frame_index, frame_count))
        # 이벤트가 발생하기 전까지는 직전 상태를 그대로 유지한다.
        labels[last_frame_index:frame_index, :] = state
        state[ACTION_INDEX[key]] = 1.0 - state[ACTION_INDEX[key]]
        last_frame_index = frame_index

        if last_frame_index >= frame_count:
            break

    if last_frame_index < frame_count:
        labels[last_frame_index:, :] = state

    return labels


def save_label_csv(sample_name: str, scenario: str, frame_count: int, labels: np.ndarray) -> Path:
    """프레임별 라벨을 CSV 파일로 저장한다."""
    output_dir = get_label_root(scenario)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{sample_name}_labels.csv"
    label_frame = pd.DataFrame(labels, columns=ACTION_KEYS)
    label_frame.insert(0, EVENT_FRAME_COLUMN, np.arange(frame_count, dtype=np.int32))
    label_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def extract_hand_landmarks(frame_paths: list[Path], output_path: Path, max_hands: int) -> Path:
    """MediaPipe Hands로 프레임별 손 랜드마크를 추출해 NPZ로 저장한다."""
    hands_solution = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=0.5,
    )

    all_keypoints: list[np.ndarray] = []

    try:
        for frame_path in frame_paths:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {frame_path}")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = hands_solution.process(image_rgb)

            # 손이 검출되지 않은 프레임도 동일한 텐서 형태를 유지한다.
            feature = np.zeros((max_hands, 21, 3), dtype=np.float32)
            if result.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks[:max_hands]):
                    for landmark_index, landmark in enumerate(hand_landmarks.landmark):
                        feature[hand_index, landmark_index, 0] = landmark.x
                        feature[hand_index, landmark_index, 1] = landmark.y
                        feature[hand_index, landmark_index, 2] = landmark.z

            all_keypoints.append(feature.reshape(-1))
    finally:
        hands_solution.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, hand_kps=np.stack(all_keypoints, axis=0))
    return output_path


def process_sample(
    scenario: str,
    sample_dir: Path,
    max_hands: int,
    skip_labels: bool,
    skip_landmarks: bool,
) -> None:
    """샘플 하나를 기준으로 라벨 CSV와 랜드마크 NPZ를 생성한다."""
    sample_name = sample_dir.name
    frame_paths = list_frame_files(sample_dir)
    if not frame_paths:
        print(f"[SKIP] {sample_name}: no frame images found")
        return

    event_csv_path = find_event_csv(sample_name, scenario)
    if not skip_labels:
        labels = build_interval_labels(len(frame_paths), event_csv_path)
        label_csv_path = save_label_csv(sample_name, scenario, len(frame_paths), labels)
        source_name = event_csv_path.name if event_csv_path is not None else "zero-filled"
        print(f"[LABEL] {sample_name}: {source_name} -> {label_csv_path}")

    if not skip_landmarks:
        landmark_path = get_landmark_root(scenario) / f"hands_{sample_name}.npz"
        extract_hand_landmarks(frame_paths, landmark_path, max_hands=max_hands)
        print(f"[LANDMARK] {sample_name}: {landmark_path}")


def main() -> None:
    """요청한 시나리오 전체를 순회하며 전처리를 실행한다."""
    args = parse_args()
    sample_dirs = find_sample_dirs(args.scenarios)

    if not sample_dirs:
        print("[ERROR] No frame directories found for the requested scenarios.")
        return

    for scenario, sample_dir in sample_dirs:
        process_sample(
            scenario=scenario,
            sample_dir=sample_dir,
            max_hands=args.max_hands,
            skip_labels=args.skip_labels,
            skip_landmarks=args.skip_landmarks,
        )


if __name__ == "__main__":
    main()
