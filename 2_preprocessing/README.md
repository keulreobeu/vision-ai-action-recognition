# 2_preprocessing

실제 실행에 사용하는 전처리 코드를 정리한 폴더입니다. 기존 실험 노트북은 [`legacy`](G:\GitProjects\sessac_project\2_preprocessing\legacy)로 분리했고, 현재 기준 실행 파일은 `.py` 스크립트 3개입니다.

## 실행 파일

- [`01_prepare_labels_and_landmarks.py`](G:\GitProjects\sessac_project\2_preprocessing\01_prepare_labels_and_landmarks.py)
  - 프레임 폴더를 읽어 프레임 단위 라벨 CSV 생성
  - MediaPipe Hands로 랜드마크 NPZ 생성
- [`02_frames_and_landmarks_to_tensors.py`](G:\GitProjects\sessac_project\2_preprocessing\02_frames_and_landmarks_to_tensors.py)
  - 프레임, 라벨 CSV, 랜드마크 NPZ를 묶어 윈도우 텐서 `.pt` 생성
- [`03_xml_to_yolo_txt.py`](G:\GitProjects\sessac_project\2_preprocessing\03_xml_to_yolo_txt.py)
  - Pascal VOC XML을 YOLO TXT로 변환

## 입력 경로

- 프레임 폴더
  - `video/video`
  - `video/missing1`
  - `video/missing2`
  - `video/idle`
- 이벤트 CSV
  - `data/out_csv/normal`
  - `data/out_csv/missing1`
  - `data/out_csv/missing2`
- XML 어노테이션
  - `yolo/label`

## 출력 경로

- 라벨 CSV
  - `data/labels/<scenario>/<sample_name>_labels.csv`
- 랜드마크 NPZ
  - `data/landmarks/<scenario>/hands_<sample_name>.npz`
- 텐서
  - `data/tensors/<scenario>/<sample_name>_windows.pt`
- YOLO TXT
  - `yolo/labels_openclose/*.txt`
  - `yolo/labels_fullempty/*.txt`

## 실행 방법

1. 라벨 CSV + 랜드마크 NPZ 생성

```bash
python 01_prepare_labels_and_landmarks.py
```

2. 윈도우 텐서 생성

```bash
python 02_frames_and_landmarks_to_tensors.py
```

3. XML to YOLO TXT

```bash
python 03_xml_to_yolo_txt.py
```

## 동작 요약

- 이벤트 CSV가 있으면 `A`, `S`, `D` 토글을 해석해 프레임 단위 라벨을 만듭니다.
- 이벤트 CSV가 없으면 전 구간을 `0,0,0` 라벨로 채웁니다.
- 랜드마크는 `hand_kps` 배열로 저장됩니다.
- 텐서 출력에는 `X_img`, `X_hand`, `y_seq`, `y_last` 등이 포함됩니다.

## 레거시 노트북

- `legacy/legacy_01_landmark_npz.ipynb`
- `legacy/legacy_02_image_to_tensor.ipynb`
- `legacy/legacy_03_xml_to_yolo_txt.ipynb`

## 필요 패키지

```bash
pip install opencv-python mediapipe numpy pandas pillow torch torchvision
```
