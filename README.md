# Vision AI Action Recognition

비전 AI 기반 행동 감지와 자동 문서화를 위한 프로젝트입니다.  
카메라로 작업 영상을 수집하고, 전처리와 모델 학습을 거쳐 예측 결과를 만들고, 마지막에 기록 문서를 자동 생성하는 흐름으로 구성되어 있습니다.

## 프로젝트 목적

이 프로젝트는 작업 영상을 사람이 직접 계속 모니터링하고 문서화해야 하는 부담을 줄이기 위해 만들었습니다.

- 영상에서 작업 행동을 감지
- 객체 상태를 함께 분석
- 이벤트를 자동으로 정리
- 최종 기록 문서를 자동 생성

## 프로젝트 흐름

1. [`1_camera`](G:\GitProjects\sessac_project\1_camera)
   - 카메라 프레임 저장
   - `A/S/D` 이벤트 수동 기록
   - 이벤트 CSV 생성
2. [`2_preprocessing`](G:\GitProjects\sessac_project\2_preprocessing)
   - 프레임 단위 라벨 CSV 생성
   - 손 랜드마크 NPZ 생성
   - 윈도우 텐서 `.pt` 생성
   - XML -> YOLO TXT 변환
3. [`3_models`](G:\GitProjects\sessac_project\3_models)
   - Google Colab 기준 행동 인식 모델 학습
   - YOLO 모델 비교 및 최종 학습
4. [`4_predict`](G:\GitProjects\sessac_project\4_predict)
   - YOLO 상태 예측
   - TCN + YOLO 결합
   - 프레임 단위 성능 평가
5. [`5_langchain`](G:\GitProjects\sessac_project\5_langchain)
   - 예측 결과를 자동 기록 CSV로 변환
   - DOCX 템플릿 자동 작성

## 로컬 / Colab 구분

- 로컬 실행 기준
  - [`1_camera`](G:\GitProjects\sessac_project\1_camera)
  - [`2_preprocessing`](G:\GitProjects\sessac_project\2_preprocessing)
  - [`4_predict`](G:\GitProjects\sessac_project\4_predict)
  - [`5_langchain`](G:\GitProjects\sessac_project\5_langchain)
- Google Colab 실행 기준
  - [`3_models`](G:\GitProjects\sessac_project\3_models)

즉 데이터 수집, 전처리, 예측, 문서화는 로컬 기준이고, 모델 학습만 Colab 기준으로 정리되어 있습니다.

## 폴더 구조

```text
sessac_project/
  1_camera/
  2_preprocessing/
  3_models/
  4_predict/
  5_langchain/
  video/
  data/
  yolo/
  docker/
  output/
  Backup/
```

## 주요 데이터 경로

- 원본/프레임 데이터: [`video`](G:\GitProjects\sessac_project\video)
- 전처리 산출물: [`data`](G:\GitProjects\sessac_project\data)
- YOLO 데이터셋 관련 파일: [`yolo`](G:\GitProjects\sessac_project\yolo)
- 최종 출력물: [`output`](G:\GitProjects\sessac_project\output)

## 단계 간 연결

- `1_camera`
  - 프레임 이미지와 이벤트 CSV 생성
- `2_preprocessing`
  - `1_camera` 결과를 읽어 라벨 CSV, 랜드마크 NPZ, 텐서 생성
- `3_models`
  - `2_preprocessing` 산출물로 행동 인식 모델 학습
  - `yolo` 데이터셋으로 YOLO 모델 학습
- `4_predict`
  - `3_models` 학습 결과와 테스트 프레임으로 예측/평가 수행
- `5_langchain`
  - `4_predict` 결과를 기록 문서 형식으로 변환

## 실행 기준

- `1_camera`, `2_preprocessing`, `4_predict`, `5_langchain`은 로컬 스크립트 기준으로 정리되어 있습니다.
- `3_models`는 리소스 문제로 Google Colab 실행을 기준으로 정리되어 있습니다.
- 기존 탐색용 노트북은 각 폴더의 `legacy/` 아래로 이동해 보관했습니다.

## 권장 환경

- Windows 또는 Linux
- Python 3.11 이상 권장
- OpenCV 사용 가능 환경
- Colab GPU 사용 가능 계정

단계별 필요 패키지는 각 폴더의 README에 따로 정리되어 있습니다.

## 권장 실행 순서

1. 카메라 데이터 수집
2. 전처리
3. Colab 모델 비교 및 최종 학습
4. 예측 및 평가
5. 문서 자동 생성

## 현재 정리 상태

- 실행용 코드와 실험용 노트북을 분리했습니다.
- 예전 노트북은 각 폴더의 `legacy/` 아래에 보관했습니다.
- 최신 기준 실행 흐름은 `.py` 스크립트 또는 새 `.ipynb` 기준으로 맞춰져 있습니다.
- 일부 README는 터미널 출력에서 한글이 깨져 보일 수 있지만, 파일 자체는 정리된 상태를 기준으로 관리하고 있습니다.

## 도커

[`docker`](G:\GitProjects\sessac_project\docker)에는 프로젝트 작업용 컨테이너 설정이 포함되어 있습니다.  
현재 구조는 문서 자동화 전용 단일 실행기보다, 전처리/예측/문서화 폴더를 함께 다루는 개발용 환경에 가깝습니다.

## 참고

- 각 단계의 자세한 사용법은 해당 폴더의 `README.md`를 참고하면 됩니다.
- 이 저장소는 실험 노트북과 실행 스크립트를 분리해, 현재는 실행용 구조를 기준으로 정리되어 있습니다.
- 학습 결과 체크포인트와 비교 결과는 Colab/Drive 기준 경로를 따릅니다.
