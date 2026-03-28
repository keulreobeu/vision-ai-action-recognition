import cv2
import os
import time
import csv
import glob
import re

"""
카메라 영상 수집 + 이벤트 플래그(키보드 A/S/D) 기록 스크립트 (프레임 저장 버전)

조작 키:
    - SPACE : 녹화 시작 / 종료 토글
    - A     : 이벤트 플래그 1
    - S     : 이벤트 플래그 2
    - D     : 이벤트 플래그 3
    - Q or ESC : 프로그램 종료 (녹화 중이면 자동으로 저장 후 종료)

추가 플래그:
    - START : 녹화 시작 시점 (flag_id = 0)
    - END   : 녹화 종료 시점 (flag_id = 9, 수동/자동/프로그램 종료 모두)

출력 구조:
    - video/<SCENARIO_DIR>/<세션폴더>/frame_000000.jpg, frame_000001.jpg, ...
    - video/<SCENARIO_DIR>/video_<SCENARIO_CODE>_<번호>_events.csv

예시:
    - SCENARIO_DIR="normal", SCENARIO_CODE="normal"
      세션 폴더: video/normal/video_normal_001/
      이벤트 CSV: video/normal/video_normal_001_events.csv
"""

# 1. 전역 변수 설정

# --- 카메라 / 영상 설정 ---
CAMERA_INDEX = 0          # 사용할 카메라 인덱스 (기본 노트북 카메라는 보통 0)
FRAME_WIDTH  = 1280       # 가로 해상도
FRAME_HEIGHT = 720        # 세로 해상도
FPS          = 30         # 프레임 레이트 (실제 저장 간격은 시스템 속도에 따라 달라질 수 있음)

# 노출값 (카메라/드라이버에 따라 무시될 수 있음)
EXPOSURE     = -9

# --- 저장 경로 설정 ---
# 최상위 비디오/프레임 폴더 이름
BASE_DIR      = "video"
# 시나리오별 폴더 이름
#   normal   : 정상
#   missing1 : 1개 누락
#   missing2 : 2개 누락
#   no_action: 행동 없음
SCENARIO_DIR  = "normal" 

# 파일 이름에 들어갈 이벤트/시나리오 약자 (원하는 문자열)
#   → video_{SCENARIO_CODE}_{번호}
SCENARIO_CODE = "normal"

# --- 자동 녹화 옵션 ---
AUTO_RECORD_SECONDS = None

# --- 프레임 저장 옵션 ---
# 저장할 이미지 포맷
IMAGE_FORMAT = "jpg"
# JPG일 때 품질
JPEG_QUALITY = 95

# --- 플래그 표시 옵션 ---
FLAG_DISPLAY_DURATION = 1.0  



# 2. 유틸리티 함수

def init_camera():
    """
    카메라(웹캠)를 초기화하고, 해상도/프레임/노출 설정을 적용한 뒤
    cv2.VideoCapture 객체를 반환한다.
    """
    # Use a Windows-specific backend only on Windows; let OpenCV choose elsewhere.
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    # 해상도 / FPS 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    # 노출값 설정
    if EXPOSURE is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    return cap


def get_output_dir():
    """
    시나리오 폴더 경로를 생성하고 반환한다.
    """
    out_dir = os.path.join(BASE_DIR, SCENARIO_DIR)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_next_index(out_dir, scenario_code):
    """
    시나리오 폴더 번호 확인
    """
    pattern = os.path.join(out_dir, f"video_{scenario_code}_*")
    paths = glob.glob(pattern)

    indices = []
    for path in paths:
        name = os.path.basename(path)
        # video_<scenario_code>_<번호>... 에서 번호만 추출
        m = re.match(rf"video_{re.escape(scenario_code)}_(\d+)", name)
        if m:
            indices.append(int(m.group(1)))

    if not indices:
        return 1
    return max(indices) + 1


def make_session_paths():
    """
    녹화 폴더/이벤트 파일 경로를 생성후 반환.

    """
    out_dir = get_output_dir()

    index = get_next_index(out_dir, SCENARIO_CODE)
    base_name = f"video_{SCENARIO_CODE}_{index:03d}"

    # 프레임 저장 폴더, 이벤트 CSV
    frames_dir = os.path.join(out_dir, base_name)
    event_path = os.path.join(out_dir, base_name + "_events.csv")

    return frames_dir, event_path


def save_frame(frames_dir, frame_idx, frame):
    """
    프레임을 이미지 파일로 저장한다.
    """
    fmt = IMAGE_FORMAT.lower()
    if fmt in ("jpg", "jpeg"):
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    elif fmt == "png":
        ext = ".png"
        # 압축 설정
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        ext = "." + fmt
        params = []

    filename = f"frame_{frame_idx:06d}{ext}"
    path = os.path.join(frames_dir, filename)

    if params:
        ok = cv2.imwrite(path, frame, params)
    else:
        ok = cv2.imwrite(path, frame)

    if not ok:
        print(f"[WARN] Failed to write frame: {path}")


def log_event(events, frame_idx, elapsed_time_sec, key_code):
    """
    이벤트 플래그(A/S/D)를 리스트(events)에 추가한다.
    """
    # 소문자 키 코드로 통일
    if key_code == ord('a'):
        flag_id, flag_key = 1, 'A'
    elif key_code == ord('s'):
        flag_id, flag_key = 2, 'S'
    elif key_code == ord('d'):
        flag_id, flag_key = 3, 'D'
    else:
        return None
    
    # 프레임 번호, 녹화 시간, 플래그 이름, 플래그 키
    events.append((frame_idx, elapsed_time_sec, flag_id, flag_key))
    return flag_id, flag_key


def save_events_csv(event_path, events):
    """
    이벤트를 CSV 파일로 저장한다.

    CSV 컬럼:
        frame_idx, time_sec, flag_id, flag_key
    """
    if not events:
        return

    with open(event_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "time_sec", "flag_id", "flag_key"])
        writer.writerows(events)


def draw_overlay(frame, recording, record_start_time,
                 last_flag_text, last_flag_time):
    """
    영상 오버레이 설정.

    recording        : 녹화 중 여부
    record_start_time: 녹화 시작 시각
    last_flag_text   : 최근에 눌린 플래그 텍스트 
    last_flag_time   : 최근 플래그가 눌린 시점
    """
    overlay = frame.copy()

    # 프레임 실제 높이/너비
    h, w = overlay.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    color_text = (255, 255, 255)  # 흰색

    # 1) 시나리오 정보 (왼쪽 상단)
    text1 = f"Scenario: {SCENARIO_DIR} ({SCENARIO_CODE})"
    cv2.putText(overlay, text1, (10, 20), font, scale, color_text, thickness, cv2.LINE_AA)

    # 2) 해상도 / FPS 정보
    text2 = f"Res: {w}x{h} (target {FRAME_WIDTH}x{FRAME_HEIGHT}) @ {FPS}fps"
    cv2.putText(overlay, text2, (10, 40), font, scale, color_text, thickness, cv2.LINE_AA)

    # 3) 녹화 상태 / 시간
    if recording and record_start_time is not None:
        elapsed = time.time() - record_start_time
        # 빨간색으로 REC 표시
        color_rec = (0, 0, 255)
        rec_text = f"REC {elapsed:5.1f}s"
        cv2.putText(overlay, rec_text, (10, 60), font, scale, color_rec, thickness + 1, cv2.LINE_AA)

        # 자동 녹화 시간 설정된 경우 남은 시간 표시
        if AUTO_RECORD_SECONDS is not None:
            remaining = max(0.0, AUTO_RECORD_SECONDS - elapsed)
            auto_text = f"AUTO STOP IN {remaining:5.1f}s"
            cv2.putText(overlay, auto_text, (10, 80), font, scale, color_rec, thickness, cv2.LINE_AA)
    else:
        # 대기 상태
        idle_text = "Press SPACE to start recording"
        cv2.putText(overlay, idle_text, (10, 60), font, scale, color_text, thickness, cv2.LINE_AA)

    # 4) 최근 플래그 표시 (화면 왼쪽 아래, 일정 시간 동안만 표시)
    now = time.time()
    if last_flag_text and (now - last_flag_time <= FLAG_DISPLAY_DURATION):
        flag_color = (0, 255, 255)  # 노란색 느낌 (BGR)
        flag_text = f"{last_flag_text} (RECORDED)"
        y_pos = h - 20  # 화면 아래쪽에서 조금 위로 올려서 표시
        cv2.putText(overlay, flag_text, (10, y_pos),
                    font, scale, flag_color, thickness + 1, cv2.LINE_AA)

    return overlay


# =========================
# 3. 녹화 제어 함수들
# =========================

def start_recording():
    """
    한 번의 녹화(세션)를 시작하기 위한 준비를 한다.

    - 세션 폴더(프레임 저장용)를 생성
    - 이벤트 CSV 경로를 정함
    - 시작 시간 기록

    반환:
        frames_dir, event_path, record_start_time, frame_idx, events
    """
    frames_dir, event_path = make_session_paths()
    os.makedirs(frames_dir, exist_ok=True)

    record_start_time = time.time()
    frame_idx = 0
    events = []

    print(f"[INFO] Recording started. Frames will be saved in: {frames_dir}")
    print(f"[INFO] Event log path: {event_path}")
    return frames_dir, event_path, record_start_time, frame_idx, events


def stop_recording(event_path, events):
    """
    녹화를 종료하고, 이벤트 CSV를 저장한다.
    """
    save_events_csv(event_path, events)
    print(f"[INFO] Recording stopped. Events saved to: {event_path}")


# =========================
# 4. 메인 루프
# =========================

def main():
    """
    전체 프로그램의 진입점.

    - 카메라 초기화
    - 프레임 읽기
    - 키보드 입력 처리 (SPACE, A/S/D, Q/ESC)
    - 녹화 상태 관리 및 이벤트 기록
    - A/S/D 플래그 입력 시 화면에 잠시 표시
    - 녹화 중일 때는 각 프레임을 이미지 파일로 저장
    - 녹화 시작/끝 시점도 START/END 플래그로 CSV에 기록
    """
    cap = init_camera()

    recording = False
    frames_dir = None
    event_path = None
    record_start_time = None
    frame_idx = 0
    events = []

    # 최근 플래그 표시용 상태 변수
    last_flag_text = ""
    last_flag_time = 0.0

    print("[INFO] Press SPACE to start/stop recording. A/S/D for flags. Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera. Exiting.")
            break

        # 상태 오버레이를 입힌 프레임 (플래그 표시 정보도 같이 전달)
        display_frame = draw_overlay(frame, recording, record_start_time,
                                     last_flag_text, last_flag_time)
        cv2.imshow("Capture", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 프로그램 종료 키 (q 또는 ESC)
        if key in (ord('q'), 27):
            if recording:
                # 녹화 중이면 먼저 END 이벤트 기록 후 종료
                if record_start_time is not None:
                    elapsed_end = time.time() - record_start_time
                else:
                    elapsed_end = 0.0

                # 마지막으로 저장된 프레임 인덱스
                last_frame_idx = max(0, frame_idx - 1)
                events.append((last_frame_idx, elapsed_end, 9, "END"))

                stop_recording(event_path, events)
                recording = False
            print("[INFO] Quit requested. Exiting.")
            break

        # SPACE : 녹화 시작/종료 토글
        if key == 32:  # Space bar
            if not recording:
                # 녹화 시작
                (frames_dir,
                 event_path,
                 record_start_time,
                 frame_idx,
                 events) = start_recording()
                recording = True

                # 녹화 시작 이벤트 기록 (START)
                # frame_idx는 0부터 시작, 시작 시점 time=0.0 으로 기록
                events.append((frame_idx, 0.0, 0, "START"))

                # 녹화 시작 시 플래그 표시 초기화
                last_flag_text = ""
                last_flag_time = 0.0
            else:
                # 녹화 종료 (수동)
                if record_start_time is not None:
                    elapsed_end = time.time() - record_start_time
                else:
                    elapsed_end = 0.0

                # 마지막으로 저장된 프레임 인덱스
                last_frame_idx = max(0, frame_idx - 1)
                events.append((last_frame_idx, elapsed_end, 9, "END"))

                stop_recording(event_path, events)
                recording = False
                frames_dir = None
                event_path = None
                record_start_time = None
                frame_idx = 0
                events = []
                # 녹화 종료 시 플래그 표시도 초기화
                last_flag_text = ""
                last_flag_time = 0.0

        # 녹화 중일 때만 실제 프레임/이벤트 기록 수행
        if recording:
            # 원본 프레임을 이미지 파일로 저장
            save_frame(frames_dir, frame_idx, frame)

            elapsed = time.time() - record_start_time

            # 자동 녹화 시간이 설정된 경우, 시간이 다 되면 자동 종료
            if (AUTO_RECORD_SECONDS is not None) and (elapsed >= AUTO_RECORD_SECONDS):
                print("[INFO] Auto stop time reached.")

                # 방금 저장한 프레임 인덱스(frame_idx)를 기준으로 END 이벤트 기록
                events.append((frame_idx, elapsed, 9, "END"))

                stop_recording(event_path, events)
                recording = False
                frames_dir = None
                event_path = None
                record_start_time = None
                frame_idx = 0
                events = []
                last_flag_text = ""
                last_flag_time = 0.0
            else:
                # A/S/D 키 입력이 있을 경우 이벤트 기록 + 화면 표시용 변수 업데이트
                if key in (ord('a'), ord('s'), ord('d')):
                    result = log_event(events, frame_idx, elapsed, key)
                    if result is not None:
                        flag_id, flag_key = result
                        last_flag_text = f"FLAG {flag_key} (ID {flag_id})"
                        last_flag_time = time.time()

                # 다음 프레임 인덱스로 증가
                frame_idx += 1

    # 리소스 정리
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
