# Legacy Model Overview

`legacy/legacy_01_models.ipynb`에서 사용한 행동 인식 모델을 현재 구조에서 한 번에 확인할 수 있도록 정리한 문서입니다.

공통 레지스트리:
- [`model_registry.py`](G:\GitProjects\sessac_project\3_models\model_registry.py)

구현 위치:
- [`behavior_modeling.py`](G:\GitProjects\sessac_project\3_models\behavior_modeling.py)

## Legacy 사용 모델

| key | model | 설정 | legacy 평균 val_acc |
| --- | --- | --- | --- |
| `legacy_tcn_v1` | TCN | `channels=(32, 32)`, `kernel_size=3`, `dropout=0.5` | `0.7152` |
| `legacy_mlp_avg_pool_v1` | MLPAvgPool | `hidden_dim=128`, `dropout=0.5` | `0.4335` |
| `legacy_cnn1d_v1` | CNN1D | `channels=(64, 128)`, `kernel_size=3`, `dropout=0.5` | `0.7253` |
| `legacy_bilstm_v1` | BiLSTM | `hidden_dim=128`, `num_layers=1`, `dropout=0.5` | `0.6836` |
| `legacy_tcn_v2` | TCN | `channels=(32, 64, 128)`, `kernel_size=3`, `dropout=0.45` | `0.7416` |
| `legacy_cnn1d_v2` | CNN1D | `channels=(32, 64, 64, 128, 128, 128)`, `kernel_size=3`, `dropout=0.45` | `0.7515` |

## 현재 비교 단계와의 관계

현재 [`02_model_comparison.ipynb`](G:\GitProjects\sessac_project\3_models\02_model_comparison.ipynb)은 기본 후보만 비교하도록 유지되어 있습니다.

- 기본 후보
  - `mlp_pool`
  - `tcn`

legacy에서 사용한 전체 후보를 확인하거나 재사용하려면 [`model_registry.py`](G:\GitProjects\sessac_project\3_models\model_registry.py)의 `get_legacy_behavior_model_specs()`를 보면 됩니다.
