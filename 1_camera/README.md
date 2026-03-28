# 1_camera

카메라 영상을 프레임 단위로 저장하고, 작업 중 발생한 이벤트를 수동 입력으로 기록하는 단계입니다.

## 개요

이 스크립트는 실시간 카메라 화면을 보여주면서 아래 작업을 수행합니다.

- 녹화 시작/종료 제어
- 프레임 이미지 저장
- `A`, `S`, `D` 이벤트 기록
- `START`, `END` 이벤트 자동 기록
- 시나리오별 출력 폴더 생성

## 실행 환경

- Python 3.x
- OpenCV

설치 예시:

```bash
pip install opencv-python
```

현재 코드는 운영체제에 따라 카메라 백엔드를 다르게 처리합니다.

- Windows: `CAP_DSHOW` 우선 사용
- Linux/Raspberry Pi: OpenCV 기본 백엔드 사용

## 실행 방법

```bash
python 01_recoding_video.py
```

## 조작 방법

- `SPACE`: 녹화 시작 / 종료
- `A`: 이벤트 플래그 1 기록
- `S`: 이벤트 플래그 2 기록
- `D`: 이벤트 플래그 3 기록
- `Q` 또는 `ESC`: 프로그램 종료

녹화가 시작되면 `START`, 종료되면 `END` 이벤트가 자동 기록됩니다.

## 출력 구조

기본 출력 경로는 `video/<SCENARIO_DIR>/`입니다.

예시:

```text
video/
  normal/
    video_normal_001/
      frame_000000.jpg
      frame_000001.jpg
      ...
    video_normal_001_events.csv
```

## CSV 형식

- `frame_idx`
- `time_sec`
- `flag_id`
- `flag_key`

예시:

```csv
frame_idx,time_sec,flag_id,flag_key
0,0.0,0,START
15,0.5,1,A
42,1.4,2,S
89,3.0,9,END
```

## 주요 설정값

`01_recoding_video.py` 상단에서 아래 값을 조정할 수 있습니다.

- `CAMERA_INDEX`
- `FRAME_WIDTH`, `FRAME_HEIGHT`
- `FPS`
- `EXPOSURE`
- `BASE_DIR`
- `SCENARIO_DIR`
- `SCENARIO_CODE`
- `AUTO_RECORD_SECONDS`
- `IMAGE_FORMAT`
- `JPEG_QUALITY`

## 참고 사항

- 일부 카메라는 FPS, 노출값 설정을 무시할 수 있습니다.
- 프리뷰 창이 활성화된 상태에서 키 입력을 받습니다.
- 영상 파일이 아니라 프레임 이미지 시퀀스를 저장합니다.
