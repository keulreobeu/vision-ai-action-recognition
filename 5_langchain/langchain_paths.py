from __future__ import annotations

"""문서 생성 단계에서 사용하는 입력/출력 파일 경로 모음."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGCHAIN_ROOT = PROJECT_ROOT / "5_langchain"

INPUT_XLSX_PATH = LANGCHAIN_ROOT / "원료_투입_기록지_자동기록_샘플 (1).xlsx"
TEMPLATE_DOCX_PATH = LANGCHAIN_ROOT / "원료_투입_기록지.docx"

OUTPUT_ROOT = LANGCHAIN_ROOT / "output"
AUTO_LOG_CSV_PATH = OUTPUT_ROOT / "auto_log.csv"
ANOMALY_LOG_CSV_PATH = OUTPUT_ROOT / "anomaly_log.csv"
OUTPUT_DOCX_PATH = OUTPUT_ROOT / "filled_record.docx"


def ensure_output_dirs() -> None:
    """문서 생성 산출물을 저장할 폴더를 준비한다."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
