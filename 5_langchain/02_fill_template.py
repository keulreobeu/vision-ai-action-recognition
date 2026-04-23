from __future__ import annotations

"""생성된 CSV를 DOCX 템플릿에 채워 최종 기록 문서를 만든다."""

import pandas as pd

from langchain_paths import ANOMALY_LOG_CSV_PATH, AUTO_LOG_CSV_PATH, OUTPUT_DOCX_PATH, TEMPLATE_DOCX_PATH, ensure_output_dirs
from document_workflow import fill_template_document


def main() -> None:
    """자동 기록표와 이상 기록표를 읽어 템플릿 문서를 채운다."""
    ensure_output_dirs()
    auto_log_df = pd.read_csv(AUTO_LOG_CSV_PATH)
    anomaly_df = pd.read_csv(ANOMALY_LOG_CSV_PATH)
    output_path = fill_template_document(TEMPLATE_DOCX_PATH, auto_log_df, anomaly_df, OUTPUT_DOCX_PATH)
    print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()
