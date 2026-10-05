"""Prepare a combined YOLO detection dataset without changing either source."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
JPG = DATA / "yolo_color-3-yolov11"
PNG = DATA / "dataset_90_5_5"
REMAPPED = DATA / "yolo_color-3-yolov11-png-class-order"
CONFIG = DATA / "targets_combined.yaml"


def main() -> None:
    if REMAPPED.exists() or CONFIG.exists():
        raise FileExistsError("Prepared dataset already exists; inspect it before rerunning")

    for old_split, new_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        image_source = JPG / old_split / "images"
        label_source = JPG / old_split / "labels"
        image_dest = REMAPPED / old_split / "images"
        label_dest = REMAPPED / old_split / "labels"
        image_dest.mkdir(parents=True)
        label_dest.mkdir(parents=True)

        for image in image_source.glob("*.jpg"):
            label = label_source / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            shutil.copy2(image, image_dest / image.name)

            rewritten = []
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5 or fields[0] not in {"0", "1"}:
                    raise ValueError(f"Unexpected YOLO label in {label}: {line}")
                fields[0] = "1" if fields[0] == "0" else "0"
                rewritten.append(" ".join(fields))
            (label_dest / label.name).write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    # Ultralytics maps each /images/ directory to its sibling /labels/ directory.
    # Keep the generated YAML portable across machines when training is
    # launched from the repository root.
    data_root = DATA.relative_to(ROOT).as_posix()
    config = f"""path: {data_root}
train:
  - dataset_90_5_5/images/train
  - yolo_color-3-yolov11-png-class-order/train/images
val:
  - dataset_90_5_5/images/val
  - yolo_color-3-yolov11-png-class-order/valid/images
test:
  - dataset_90_5_5/images/test
  - yolo_color-3-yolov11-png-class-order/test/images
names:
  0: enemy
  1: ally
"""
    CONFIG.write_text(config, encoding="utf-8")
    print(f"Prepared JPG labels and combined configuration: {CONFIG}")


if __name__ == "__main__":
    main()
