"""Train and evaluate a two-class target detector on the combined dataset."""

from pathlib import Path
import os

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_CONFIG = ROOT / "data" / "targets_combined.yaml"
RUNS = ROOT / "runs" / "detect"
RUN_NAME = "targets_combined_v2"


def main() -> None:
    # Resolve the relative paths in targets_combined.yaml from this checkout,
    # regardless of the caller's current working directory.
    os.chdir(ROOT)

    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Prepare the dataset first: {DATA_CONFIG}")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_CONFIG),
        epochs=80,
        imgsz=960,
        batch=8,
        patience=20,
        device=0,
        workers=4,
        hsv_h=0.0,
        hsv_s=0.15,
        hsv_v=0.15,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=10,
        save_period=10,
        seed=42,
        deterministic=True,
        plots=True,
        project=str(RUNS),
        name=RUN_NAME,
        exist_ok=False,
    )

    best = RUNS / RUN_NAME / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished without best.pt: {best}")

    YOLO(str(best)).val(data=str(DATA_CONFIG), split="test", imgsz=960, device=0)
    print(f"BEST_WEIGHTS={best}")


if __name__ == "__main__":
    main()
