"""도커 환경이 정상 동작하는지 간단히 확인하는 테스트 스크립트."""

from pathlib import Path

def main():
    """입력 파일 목록을 읽어 결과 텍스트 파일로 남긴다."""
    input_dir = Path("/workspace/input")
    output_dir = Path("/workspace/output")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*"))

    with open(output_dir / "result.txt", "w", encoding="utf-8") as f:
        f.write("문서 자동화 테스트 성공\n")
        f.write(f"입력 파일 개수: {len(files)}\n")
        for file in files:
            f.write(f"- {file.name}\n")

    print("작업 완료: output/result.txt 생성")

if __name__ == "__main__":
    main()
