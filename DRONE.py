import argparse
from pathlib import Path
import time
import cv2
from ultralytics import YOLO


# ---------- Drawing helpers ----------
def draw_box_label(img, xyxy, label, conf, color=(255, 0, 255)):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, txt, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, fps, unique_people):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total unique people: {unique_people}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


# ---------- Main pipeline ----------
def run(source, model_path, conf, iou, people_only, save_path, show):
    # ép kiểu để tránh lỗi float32
    conf = float(conf)
    iou = float(iou)

    device_hint = "mps"  # macOS Apple Silicon (M1/M2)
    model = YOLO(model_path)

    writer = None
    unique_person_ids = set()
    last_t = time.time()
    fps = 0.0

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLOv8 tracking
        results = model.track(
            frame,
            conf=conf,
            iou=iou,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            stream=True,
            classes=[0] if people_only else None  # class 0 = person
        )

        for r in results:
            names = model.model.names if hasattr(model, "model") else r.names
            if r.boxes is not None and len(r.boxes) > 0:
                xyxys = r.boxes.xyxy.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                ids = r.boxes.id
                ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(clss)

                for (xyxy, c, conf_score, tid) in zip(xyxys, clss, confs, ids):
                    label = (
                        names.get(c, str(c))
                        if isinstance(names, dict)
                        else (names[c] if c < len(names) else str(c))
                    )
                    # Count unique people ever seen
                    if label == "person" and tid is not None:
                        unique_person_ids.add(int(tid))
                    draw_box_label(frame, xyxy, label, conf_score)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - last_t)) if now > last_t else fps
        last_t = now
        draw_hud(frame, fps, len(unique_person_ids))

        # Init writer
        if save_path and writer is None:
            h, w = frame.shape[:2]
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(save_path, fourcc, max(10.0, fps or 25.0), (w, h))

        if writer is not None:
            writer.write(frame)

        if show:
            cv2.imshow("YOLOv8 Real-time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="Real-time YOLOv8 detection on macOS")
    ap.add_argument("--source", default="0",
                    help="Camera index ('0') or path to video/image (e.g. DRONEVIDEO.MP4)")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="Path or name of YOLOv8 model (yolov8n.pt, yolov8s.pt, custom.pt)")
    ap.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    ap.add_argument("--people-only", action="store_true",
                    help="Only detect persons (class 0)")
    ap.add_argument("--save", default="", help="Output video path (e.g. out/result.mp4)")
    ap.add_argument("--show", action="store_true", help="Show window in realtime")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        people_only=args.people_only,
        save_path=args.save if args.save else None,
        show=args.show or (args.source != "0")
    )
