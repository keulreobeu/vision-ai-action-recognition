from __future__ import annotations

"""행동 인식 모델 후보와 레거시 실험 정보를 정리한 레지스트리."""

from dataclasses import dataclass
from typing import Callable

from behavior_modeling import (
    BiLSTMClassifier,
    MLPAvgPoolClassifier,
    MLPTemporalPoolingClassifier,
    SimpleCNN1DClassifier,
    TCNClassifier,
)


@dataclass(frozen=True)
class LegacyBehaviorModelSpec:
    """레거시 노트북에서 가져온 모델 비교 정보를 담는 구조체."""
    key: str
    display_name: str
    family: str
    source_notebook: str
    source_section: str
    mean_val_acc: float
    builder: Callable[[int, int], object]
    notes: str


def get_current_behavior_model_builders() -> dict[str, Callable[[int, int], object]]:
    """현재 파이썬 워크플로우에서 바로 재사용하는 모델 빌더를 반환한다."""
    return {
        "mlp_pool": lambda input_dim, num_classes: MLPTemporalPoolingClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
        ),
        "tcn": lambda input_dim, num_classes: TCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            channels=(64, 64),
        ),
    }


def get_legacy_behavior_model_specs() -> list[LegacyBehaviorModelSpec]:
    """과거 노트북 실험 결과를 코드에서 참조할 수 있게 정리한다."""
    return [
        LegacyBehaviorModelSpec(
            key="legacy_tcn_v1",
            display_name="Legacy TCN v1",
            family="TCN",
            source_notebook="legacy/legacy_01_models.ipynb",
            source_section="initial_tcn",
            mean_val_acc=0.7152444265735853,
            builder=lambda input_dim, num_classes: TCNClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                channels=(32, 32),
                kernel_size=3,
                dropout=0.5,
            ),
            notes="channels=(32, 32), kernel_size=3, dropout=0.5",
        ),
        LegacyBehaviorModelSpec(
            key="legacy_mlp_avg_pool_v1",
            display_name="Legacy MLP AvgPool v1",
            family="MLP",
            source_notebook="legacy/legacy_01_models.ipynb",
            source_section="mlp_avg_pool",
            mean_val_acc=0.43346148537249135,
            builder=lambda input_dim, num_classes: MLPAvgPoolClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=128,
                dropout=0.5,
            ),
            notes="hidden_dim=128, dropout=0.5",
        ),
        LegacyBehaviorModelSpec(
            key="legacy_cnn1d_v1",
            display_name="Legacy CNN1D v1",
            family="CNN1D",
            source_notebook="legacy/legacy_01_models.ipynb",
            source_section="cnn1d_v1",
            mean_val_acc=0.725310767401116,
            builder=lambda input_dim, num_classes: SimpleCNN1DClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                channels=(64, 128),
                kernel_size=3,
                dropout=0.5,
            ),
            notes="channels=(64, 128), kernel_size=3, dropout=0.5",
        ),
        LegacyBehaviorModelSpec(
            key="legacy_bilstm_v1",
            display_name="Legacy BiLSTM v1",
            family="BiLSTM",
            source_notebook="legacy/legacy_01_models.ipynb",
            source_section="bilstm",
            mean_val_acc=0.6835800357402041,
            builder=lambda input_dim, num_classes: BiLSTMClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=128,
                num_layers=1,
                dropout=0.5,
            ),
            notes="hidden_dim=128, num_layers=1, bidirectional=True, dropout=0.5",
        ),
        LegacyBehaviorModelSpec(
            key="legacy_tcn_v2",
            display_name="Legacy TCN v2",
            family="TCN",
            source_notebook="legacy/legacy_01_models.ipynb",
            source_section="improved_tcn",
            mean_val_acc=0.7415547266286431,
            builder=lambda input_dim, num_classes: TCNClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                channels=(32, 64, 128),
                kernel_size=3,
                dropout=0.45,
            ),
            notes="channels=(32, 64, 128), kernel_size=3, dropout=0.45",
        ),
        LegacyBehaviorModelSpec(
            key="legacy_cnn1d_v2",
            display_name="Legacy CNN1D v2",
            family="CNN1D",
            source_notebook="legacy/legacy_01_models.ipynb",
            source_section="improved_cnn1d",
            mean_val_acc=0.7514829644221491,
            builder=lambda input_dim, num_classes: SimpleCNN1DClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                channels=(32, 64, 64, 128, 128, 128),
                kernel_size=3,
                dropout=0.45,
            ),
            notes="channels=(32, 64, 64, 128, 128, 128), kernel_size=3, dropout=0.45",
        ),
    ]
