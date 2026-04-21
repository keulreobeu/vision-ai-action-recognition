# 4_predict

학습된 모델 산출물을 이용해 테스트 프레임을 예측하고, TCN과 YOLO 결과를 결합한 뒤 프레임 단위 성능을 확인하는 단계입니다. 기존 노트북은 [`legacy`](G:\GitProjects\sessac_project\4_predict\legacy)로 옮기고, 현재는 실행용 스크립트 기준으로 정리했습니다.

## 구성 파일

- [`predict_paths.py`](G:\GitProjects\sessac_project\4_predict\predict_paths.py)
- [`predict_workflow.py`](G:\GitProjects\sessac_project\4_predict\predict_workflow.py)
- [`01_run_yolo_state_prediction.py`](G:\GitProjects\sessac_project\4_predict\01_run_yolo_state_prediction.py)
- [`02_fuse_tcn_and_yolo.py`](G:\GitProjects\sessac_project\4_predict\02_fuse_tcn_and_yolo.py)
- [`03_score_predictions.py`](G:\GitProjects\sessac_project\4_predict\03_score_predictions.py)

## 기본 경로

- 테스트 프레임: `yolo/test_video/`
- YOLO 가중치: `yolo/best_openclose.pt`, `yolo/best_fullempty.pt`
- TCN 예측 CSV 탐색:
  - `4_predict/output/tcn_predictions/`
  - `yolo/test_video/out_TCN/`
- GT 라벨 CSV 탐색:
  - `data/labels/`
  - `Backup/old/test_data/test_in_model/test_csv/`

출력:

- `4_predict/output/yolo_states/`
- `4_predict/output/fused_frames/`
- `4_predict/output/events/`
- `4_predict/output/metrics/`

## 실행 순서

1. YOLO 상태 생성

```powershell
python 4_predict/01_run_yolo_state_prediction.py
```

2. TCN + YOLO 결합

```powershell
python 4_predict/02_fuse_tcn_and_yolo.py
```

3. 프레임 단위 성능 평가

```powershell
python 4_predict/03_score_predictions.py
```

## 입출력 형식

YOLO 상태 CSV:

- `video_name`, `frame_idx`, `frame_name`, `box_count`, `open_count`, `closed_count`, `full_count`, `empty_count`

결합 프레임 CSV:

- `A`, `S`, `D`
- YOLO 상태 컬럼
- `tcn_label`, `yolo_A_like`, `yolo_S_like`, `yolo_D_like`, `fused_label_raw`, `fused_label`

이벤트 CSV:

- `frame_idx`, `flag_id`, `flag_key`

성능 CSV:

- `exact_frame_acc`
- `micro_precision`, `micro_recall`, `micro_f1`, `micro_acc`
- 클래스별 `A/S/D` precision, recall, f1, acc, support



## 주의 사항

- `03_score_predictions.py`는 프레임 단위 GT 라벨 CSV를 기준으로 평가합니다.
- TCN 예측 CSV는 `A`, `S`, `D` 컬럼이 있어야 합니다.
- 이 환경에서는 `python` 실행기 문제로 실제 런타임 검증은 하지 못했습니다.
