"""Pascal VOC XML 어노테이션을 두 종류의 YOLO 라벨로 변환한다."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.etree.ElementTree import ParseError

from preprocessing_paths import (
    YOLO_FULL_EMPTY_ROOT,
    YOLO_OPEN_CLOSE_ROOT,
    YOLO_XML_ROOT,
)


OPEN_CLOSE_CLASS_MAP = {
    "open_empty": 0,
    "open_full": 0,
    "close_full": 1,
    "close_empty": 1,
}

FULL_EMPTY_CLASS_MAP = {
    "open_empty": 0,
    "close_empty": 0,
    "open_full": 1,
    "close_full": 1,
}


def parse_args() -> argparse.Namespace:
    """XML 입력 경로와 출력 경로를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML annotations into two YOLO label sets.",
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        default=YOLO_XML_ROOT,
        help="Root directory that contains Pascal VOC XML files.",
    )
    parser.add_argument(
        "--open-close-root",
        type=Path,
        default=YOLO_OPEN_CLOSE_ROOT,
        help="Output directory for open/close labels.",
    )
    parser.add_argument(
        "--full-empty-root",
        type=Path,
        default=YOLO_FULL_EMPTY_ROOT,
        help="Output directory for full/empty labels.",
    )
    return parser.parse_args()


def voc_to_yolo_bbox(image_size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """VOC 박스를 YOLO 형식의 정규화 좌표로 변환한다."""
    width, height = image_size
    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert_single_xml(
    xml_path: Path,
    open_close_root: Path,
    full_empty_root: Path,
) -> None:
    """XML 한 장을 읽어 open/close, full/empty 라벨 파일을 동시에 만든다."""
    try:
        tree = ET.parse(xml_path)
    except ParseError as error:
        print(f"[SKIP] Parse error in {xml_path}: {error}")
        return

    root = tree.getroot()
    size_node = root.find("size")
    if size_node is None:
        print(f"[SKIP] Missing size node in {xml_path}")
        return

    image_width = int(size_node.findtext("width", default="0"))
    image_height = int(size_node.findtext("height", default="0"))
    if image_width <= 0 or image_height <= 0:
        print(f"[SKIP] Invalid image size in {xml_path}")
        return

    open_close_lines: list[str] = []
    full_empty_lines: list[str] = []

    for object_node in root.findall("object"):
        class_name = object_node.findtext("name", default="").strip()
        if class_name not in OPEN_CLOSE_CLASS_MAP or class_name not in FULL_EMPTY_CLASS_MAP:
            print(f"[SKIP] Unknown class '{class_name}' in {xml_path}")
            continue

        bbox_node = object_node.find("bndbox")
        if bbox_node is None:
            print(f"[SKIP] Missing bndbox in {xml_path}")
            continue

        xmin = float(bbox_node.findtext("xmin", default="0"))
        ymin = float(bbox_node.findtext("ymin", default="0"))
        xmax = float(bbox_node.findtext("xmax", default="0"))
        ymax = float(bbox_node.findtext("ymax", default="0"))

        x_center, y_center, box_width, box_height = voc_to_yolo_bbox(
            (image_width, image_height),
            (xmin, ymin, xmax, ymax),
        )
        open_close_lines.append(
            f"{OPEN_CLOSE_CLASS_MAP[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
        full_empty_lines.append(
            f"{FULL_EMPTY_CLASS_MAP[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    open_close_root.mkdir(parents=True, exist_ok=True)
    full_empty_root.mkdir(parents=True, exist_ok=True)

    base_name = xml_path.stem
    (open_close_root / f"{base_name}.txt").write_text("\n".join(open_close_lines), encoding="utf-8")
    (full_empty_root / f"{base_name}.txt").write_text("\n".join(full_empty_lines), encoding="utf-8")
    print(f"[YOLO] {xml_path.name}")


def main() -> None:
    """XML 전체를 순회하며 YOLO TXT 파일을 생성한다."""
    args = parse_args()
    xml_files = sorted(args.xml_root.rglob("*.xml"))
    if not xml_files:
        print(f"[ERROR] No XML files found under {args.xml_root}")
        return

    for xml_file in xml_files:
        convert_single_xml(
            xml_path=xml_file,
            open_close_root=args.open_close_root,
            full_empty_root=args.full_empty_root,
        )


if __name__ == "__main__":
    main()
