import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / "best_yaw_cnn.pth"
YOLO_MODEL_PATH = BASE_DIR / "yolo_target.pt"
SCALER_PATH = BASE_DIR / "yaw_scaler.json"

YAW_PIN = 18
PITCH_PIN = 13
FIRE_PIN = 12

CENTER_YAW = 50.0
FIXED_PITCH = 90.0
YAW_SERVO_MIN = 32.0
YAW_SERVO_MAX = 68.0
YAW_DIRECTION = 1.0
BARREL_ALIGNMENT_OFFSET = -2.0
SERVO_STEP = 1.0
SERVO_DELAY = 0.03

FIRE_READY_ANGLE = 90.0
FIRE_SHOOT_ANGLE = 170.0
AIM_SETTLE_DELAY = 0.5
FIRE_HOLD_DELAY = 0.4
FIRE_RESET_DELAY = 0.5
FIRE_COOLDOWN = 1.0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMAGE_SIZE = 224
WINDOW_NAME = "Yaw CNN Inference"
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
YOLO_CONFIDENCE = 0.5
DEVICE = torch.device("cpu")


class YawCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def load_models():
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

    try:
        checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(CNN_MODEL_PATH, map_location=DEVICE)

    model = YawCNN().to(DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.eval()

    if SCALER_PATH.exists():
        with SCALER_PATH.open("r", encoding="utf-8") as file:
            scaler = json.load(file)
        yaw_min = float(scaler["yaw_min"])
        yaw_max = float(scaler["yaw_max"])
    elif isinstance(checkpoint, dict) and {"yaw_min", "yaw_max"} <= checkpoint.keys():
        yaw_min = float(checkpoint["yaw_min"])
        yaw_max = float(checkpoint["yaw_max"])
    else:
        raise FileNotFoundError(
            "yaw_scaler.json was not found, and the checkpoint does not contain yaw_min/yaw_max."
        )

    return model, YOLO(str(YOLO_MODEL_PATH)), yaw_min, yaw_max


def prepare_frame(frame):
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    resized = cv2.resize(rotated, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def detect_target(frame, yolo_model):
    result = yolo_model.predict(source=frame, conf=YOLO_CONFIDENCE, verbose=False)[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None, None, None

    confidences = boxes.conf.detach().cpu().numpy()
    best_index = int(np.argmax(confidences))
    x1, y1, x2, y2 = np.rint(boxes.xyxy[best_index].detach().cpu().numpy()).astype(int)

    x1 = int(np.clip(x1, 0, IMAGE_SIZE - 1))
    y1 = int(np.clip(y1, 0, IMAGE_SIZE - 1))
    x2 = int(np.clip(x2, 0, IMAGE_SIZE - 1))
    y2 = int(np.clip(y2, 0, IMAGE_SIZE - 1))

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask, (x1, y1, x2, y2), float(confidences[best_index])


def predict_yaw(mask, model, yaw_min, yaw_max):
    tensor = (
        torch.from_numpy(mask)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        normalized_yaw = model(tensor).item()

    return normalized_yaw * (yaw_max - yaw_min) + yaw_min


def draw_detection(frame, box, confidence):
    if box is None:
        cv2.putText(
            frame,
            "NO TARGET",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return

    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"TARGET {confidence:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def set_servo(pi, pin, angle):
    angle = float(np.clip(angle, 0.0, 180.0))
    pi.set_servo_pulsewidth(pin, 500.0 + (angle / 180.0) * 2000.0)


def move_yaw(pi, current_angle, target_angle):
    target_angle = float(np.clip(target_angle, YAW_SERVO_MIN, YAW_SERVO_MAX))

    while abs(target_angle - current_angle) > 0.05:
        difference = target_angle - current_angle
        step = min(SERVO_STEP, abs(difference))
        current_angle += step if difference > 0 else -step
        set_servo(pi, YAW_PIN, current_angle)
        time.sleep(SERVO_DELAY)

    set_servo(pi, YAW_PIN, target_angle)
    return target_angle


def fire(pi):
    set_servo(pi, FIRE_PIN, FIRE_SHOOT_ANGLE)
    time.sleep(FIRE_HOLD_DELAY)
    set_servo(pi, FIRE_PIN, FIRE_READY_ANGLE)
    time.sleep(FIRE_RESET_DELAY)


def stop_hardware(pi, camera):
    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass

    cv2.destroyAllWindows()

    for pin in (YAW_PIN, PITCH_PIN, FIRE_PIN):
        pi.set_servo_pulsewidth(pin, 0)

    pi.stop()


def main():
    import pigpio
    from picamera2 import Picamera2

    cnn_model, yolo_model, yaw_min, yaw_max = load_models()
    pi = pigpio.pi()

    if not pi.connected:
        raise RuntimeError("pigpio is not connected. Run: sudo pigpiod")

    camera = None
    current_yaw = CENTER_YAW
    last_fire_time = 0.0

    try:
        set_servo(pi, YAW_PIN, current_yaw)
        set_servo(pi, PITCH_PIN, FIXED_PITCH)
        set_servo(pi, FIRE_PIN, FIRE_READY_ANGLE)
        time.sleep(0.7)

        camera = Picamera2()
        camera.configure(
            camera.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
            )
        )
        camera.start()
        time.sleep(1.0)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

        print("SPACE: aim and fire | R: center yaw | ESC/Q: quit")

        while True:
            model_input = prepare_frame(camera.capture_array())
            mask, box, confidence = detect_target(model_input, yolo_model)

            preview = model_input.copy()
            draw_detection(preview, box, confidence)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("r"):
                current_yaw = move_yaw(pi, current_yaw, CENTER_YAW)
                continue

            if key != ord(" "):
                continue

            elapsed = time.monotonic() - last_fire_time
            if elapsed < FIRE_COOLDOWN:
                continue

            if mask is None:
                print("Target not detected.")
                continue

            yaw_offset = predict_yaw(mask, cnn_model, yaw_min, yaw_max)
            target_yaw = current_yaw + (YAW_DIRECTION * yaw_offset) + BARREL_ALIGNMENT_OFFSET
            current_yaw = move_yaw(pi, current_yaw, target_yaw)

            time.sleep(AIM_SETTLE_DELAY)
            fire(pi)
            last_fire_time = time.monotonic()

    except KeyboardInterrupt:
        pass
    finally:
        stop_hardware(pi, camera)


if __name__ == "__main__":
    main()
