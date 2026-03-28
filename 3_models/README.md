# 3_models

이 폴더는 로컬 실행용이 아니라 Google Colab 실행을 전제로 정리한 모델 학습 단계입니다. 기존 실험 노트북은 [`legacy`](G:\GitProjects\sessac_project\3_models\legacy)로 분리했고, 현재 작업 기준 노트북은 3단계로 구성했습니다.

## 현재 사용 노트북

- [`01_colab_exploration.ipynb`](G:\GitProjects\sessac_project\3_models\01_colab_exploration.ipynb)
  - 최신 전처리 산출물 경로 확인
  - 행동 인식 샘플 분포, fold 구성 확인
  - YOLO 데이터셋 샘플 수 확인
- [`02_model_comparison.ipynb`](G:\GitProjects\sessac_project\3_models\02_model_comparison.ipynb)
  - 현재 후보 + legacy 모델 목록 비교
  - YOLO 백본 후보 비교
- [`03_final_model_training.ipynb`](G:\GitProjects\sessac_project\3_models\03_final_model_training.ipynb)
  - 최종 행동 인식 모델 학습 및 저장
  - 최종 YOLO 모델 학습 및 저장

## 입력 경로

행동 인식 최신 입력:

- `data/labels/<scenario>/<sample_name>_labels.csv`
- `data/landmarks/<scenario>/hands_<sample_name>.npz`

행동 인식 레거시 호환 입력:

- `data/out_csv/<scenario>/*_lange.csv`
- `data/out_npz/<scenario>/hands_*.npz`

YOLO 입력:

- `yolo/imege`
- `yolo/labels_openclose`
- `yolo/labels_fullempty`

## 공통 모듈

- [`colab_paths.py`](G:\GitProjects\sessac_project\3_models\colab_paths.py)
- [`behavior_modeling.py`](G:\GitProjects\sessac_project\3_models\behavior_modeling.py)
- [`model_registry.py`](G:\GitProjects\sessac_project\3_models\model_registry.py)
- [`yolo_workflow.py`](G:\GitProjects\sessac_project\3_models\yolo_workflow.py)
- [`LEGACY_MODEL_OVERVIEW.md`](G:\GitProjects\sessac_project\3_models\LEGACY_MODEL_OVERVIEW.md)

## 추천 실행 순서

1. `01_colab_exploration.ipynb`
2. `02_model_comparison.ipynb`
3. `03_final_model_training.ipynb`

## Colab 기본 설정

기본 프로젝트 루트 예시:

```python
PROJECT_DIR = Path('/content/drive/MyDrive/sessac_project')
```

## 주요 산출물

기본 저장 위치:

```text
/content/drive/MyDrive/sessac_project_artifacts/
```

비교 단계:

- `model_comparison/behavior_model_catalog.csv`
- `model_comparison/behavior_model_comparison.csv`
- `model_comparison/yolo_model_comparison.csv`

최종 학습 단계:

- `final_training/behavior/behavior_tcn_final.pt`
- `final_training/behavior/behavior_tcn_history.csv`
- `final_training/behavior/behavior_tcn_summary.json`
- `final_training/yolo/runs/open_close_final/weights/best.pt`
- `final_training/yolo/runs/full_empty_final/weights/best.pt`
- `final_training/final_summary.json`

## 레거시 노트북

- `legacy/legacy_00_colab_environment.ipynb`
- `legacy/legacy_01_models.ipynb`
- `legacy/legacy_02_yolo_bbox.ipynb`

## Colab 패키지

```bash
pip install ultralytics pyyaml
```
