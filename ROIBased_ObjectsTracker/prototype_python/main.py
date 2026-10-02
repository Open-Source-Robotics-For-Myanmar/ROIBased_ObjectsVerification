#!/usr/bin/env python3
"""
Manual ROI Tracker + Kalman Filter
----------------------------------
- Camera OR video file (--video)
- Manual ROI selection (no detection)
- Tracker: CSRT / KCF / MOSSE
- Kalman filter smooths (cx, cy)
- Drag with the mouse to select or re-select the target
"""

import argparse
import sys
import time
import cv2
import numpy as np

from kalman_tracker import KalmanFilter2D


# ----------------------------------------------------------------------
# Tracker factory (handles old/new OpenCV API)
# ----------------------------------------------------------------------
def create_tracker(name: str):
    name = name.upper()
    if name == "CSRT":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
    elif name == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
    elif name == "MOSSE":
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
            return cv2.legacy.TrackerMOSSE_create()
        if hasattr(cv2, "TrackerMOSSE_create"):
            return cv2.TrackerMOSSE_create()
    raise ValueError(f"Tracker '{name}' is not available in this OpenCV build.")


# ----------------------------------------------------------------------
# Wrapper: tracker + kalman
# ----------------------------------------------------------------------
class ManualROITracker:
    def __init__(self, frame, bbox, tracker_name="CSRT",
                 process_noise=1e-2, measurement_noise=5e-1):
        self.tracker = create_tracker(tracker_name)
        self.tracker.init(frame, bbox)
        self.bbox = bbox

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0

        self.kf = KalmanFilter2D(dt=1.0,
                                 process_noise=process_noise,
                                 measurement_noise=measurement_noise)
        self.kf.init(cx, cy)

        # Smoothed / predicted positions
        self.smooth_cx, self.smooth_cy = cx, cy
        self.pred_cx,   self.pred_cy   = cx, cy

        # Raw measurement (from tracker)
        self.meas_cx, self.meas_cy = cx, cy

        self.ok = True

    def update(self, frame):
        self.ok, self.bbox = self.tracker.update(frame)
        pred = self.kf.predict()
        self.pred_cx, self.pred_cy = float(pred[0, 0]), float(pred[1, 0])

        if self.ok:
            x, y, w, h = self.bbox
            self.meas_cx = x + w / 2.0
            self.meas_cy = y + h / 2.0
            corr = self.kf.correct(self.meas_cx, self.meas_cy)
            self.smooth_cx, self.smooth_cy = float(corr[0, 0]), float(corr[1, 0])
        else:
            self.meas_cx, self.meas_cy = None, None
            self.smooth_cx, self.smooth_cy = self.pred_cx, self.pred_cy
        return self.ok


class ROISelector:
    def __init__(self):
        self.dragging = False
        self.start = None
        self.end = None
        self.selection_ready = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.dragging = True
            self.selection_ready = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = (x, y)
            self.dragging = False
            self.selection_ready = True

    def draw(self, frame):
        if self.dragging and self.start and self.end:
            x1, y1 = self.start
            x2, y2 = self.end
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def take_selection(self, frame_shape):
        if not self.selection_ready or self.start is None or self.end is None:
            return None

        self.selection_ready = False
        height, width = frame_shape[:2]
        x1 = max(0, min(self.start[0], self.end[0]))
        y1 = max(0, min(self.start[1], self.end[1]))
        x2 = min(width, max(self.start[0], self.end[0]))
        y2 = min(height, max(self.start[1], self.end[1]))
        self.start = None
        self.end = None
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return (x1, y1, x2 - x1, y2 - y1)


# ----------------------------------------------------------------------
# UI drawing
# ----------------------------------------------------------------------
def draw_tracking_box(frame, tracker):
    if tracker is not None:
        x, y, bw, bh = [int(v) for v in tracker.bbox]
        box_color = (0, 255, 0) if tracker.ok else (0, 0, 255)
        sx, sy = int(tracker.smooth_cx), int(tracker.smooth_cy)
        corner = max(8, min(24, min(bw, bh) // 4))
        thickness = 4
        for x1, y1, dx, dy in (
            (x, y, 1, 1), (x + bw, y, -1, 1),
            (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1),
        ):
            cv2.line(frame, (x1, y1), (x1 + dx * corner, y1), box_color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + dy * corner), box_color, thickness)
        scope_radius = max(10, min(18, min(bw, bh) // 5))
        scope_color = (0, 0, 255) if not tracker.ok else box_color
        cv2.circle(frame, (sx, sy), scope_radius, scope_color, 2)
        cv2.drawMarker(frame, (sx, sy), scope_color, cv2.MARKER_CROSS,
                       scope_radius * 2 + 12, 2)
        cv2.circle(frame, (sx, sy), 2, scope_color, -1)
        if not tracker.ok:
            cv2.putText(frame, "TARGET LOST - drag to reselect", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Manual ROI Tracker + Kalman Filter")
    p.add_argument("--video", type=str, default=None,
                   help="Path to video file (if not set, uses camera)")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default: 0)")
    p.add_argument("--tracker", type=str, default="CSRT",
                   choices=["CSRT", "KCF", "MOSSE"])
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--loop", action="store_true",
                   help="Loop video file when finished")
    return p.parse_args()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source_name = f"camera #{args.camera}"

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source_name}")
        sys.exit(1)

    win_main = "Manual ROI Tracker (Kalman)"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    selector = ROISelector()
    cv2.setMouseCallback(win_main, selector.on_mouse)

    print("=" * 44)
    print("  Manual ROI Tracker + Kalman Filter")
    print("=" * 44)
    print("  Mouse drag : select or re-select target")
    print("  p     : pause / resume")
    print("  q     : quit")
    print(f"  Source : {source_name}")
    print(f"  Tracker: {args.tracker}")
    print("=" * 44)

    tracker   = None
    paused    = False
    frame     = None
    fps_frames = 0
    fps_elapsed = 0.0
    fps_value = 0.0

    while True:
        frame_started = None
        if not paused:
            frame_started = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                if args.video and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[INFO] End of stream.")
                break

            if tracker is not None:
                tracker.update(frame)

        if frame is None:
            continue

        vis = frame.copy()
        selector.draw(vis)
        draw_tracking_box(vis, tracker)
        if tracker is None:
            cv2.putText(vis, "Drag mouse around target to select", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if frame_started is not None:
            fps_frames += 1
            fps_elapsed += time.perf_counter() - frame_started
            if fps_elapsed >= 1.0:
                fps_value = fps_frames / fps_elapsed
                fps_frames = 0
                fps_elapsed = 0.0
        if fps_value > 0.0:
            fps_label = f"FPS: {fps_value:.1f}"
            fps_position = (max(10, vis.shape[1] - 150), 30)
            cv2.putText(vis, fps_label, fps_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(vis, fps_label, fps_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow(win_main, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")

        roi = selector.take_selection(frame.shape)
        if roi is not None:
            tracker = ManualROITracker(frame, roi, tracker_name=args.tracker)
            print(f"[INFO] Target locked: bbox={roi}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()