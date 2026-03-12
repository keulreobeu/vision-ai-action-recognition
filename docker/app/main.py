from pathlib import Path

def main():
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