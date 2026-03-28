# 5_langchain

예측 결과를 기록 문서로 변환하는 자동 문서화 단계입니다. 기존 실험 노트북은 [`legacy`](G:\GitProjects\sessac_project\5_langchain\legacy)로 옮겼고, 현재는 실행용 스크립트 기준으로 정리했습니다.

## 구성 파일

- [`langchain_paths.py`](G:\GitProjects\sessac_project\5_langchain\langchain_paths.py)
- [`document_workflow.py`](G:\GitProjects\sessac_project\5_langchain\document_workflow.py)
- [`01_generate_logs.py`](G:\GitProjects\sessac_project\5_langchain\01_generate_logs.py)
- [`02_fill_template.py`](G:\GitProjects\sessac_project\5_langchain\02_fill_template.py)

## 기본 입력 파일

- 입력 XLSX: [`원료_투입_기록지_자동기록_샘플 (1).xlsx`](G:\GitProjects\sessac_project\5_langchain\원료_투입_기록지_자동기록_샘플%20(1).xlsx)
- 템플릿 DOCX: [`원료_투입_기록지.docx`](G:\GitProjects\sessac_project\5_langchain\원료_투입_기록지.docx)

필수 컬럼:

- `time_sec`
- `flag_id`

`flag_id`는 `1/A`, `2/S`, `3/D`를 허용합니다.

## 출력 경로

모든 결과는 [`output`](G:\GitProjects\sessac_project\5_langchain\output) 아래에 저장됩니다.

- `auto_log.csv`
- `anomaly_log.csv`
- `filled_record.docx`

## 실행 순서

1. 자동 기록 / 이상 이벤트 생성

```powershell
python 5_langchain/01_generate_logs.py
```

2. 템플릿 문서 작성

```powershell
python 5_langchain/02_fill_template.py
```

## LLM 사용 방식

현재 코드는 Gemini를 선택적으로 사용합니다.

- `GEMINI_API_KEY`가 설정되어 있고 `google-generativeai`를 import할 수 있으면 이상 이벤트 설명 생성에 Gemini를 사용합니다.
- 키가 없거나 호출에 실패하면 중단하지 않고 규칙 기반 기본 문구로 대체합니다.

## 정리한 내용

- `C:\Users\user\Downloads\...` 절대 경로 제거
- 노트북 셀 순서 의존성 제거
- 2단계 스크립트로 분리
- 출력 경로를 `output/` 아래로 통일

## 주의 사항

- 이 환경에서는 `python` 실행기 문제로 실제 런타임 검증은 하지 못했습니다.
- 템플릿 DOCX 표 구조가 크게 바뀌면 테이블 인식 로직도 함께 조정해야 합니다.
