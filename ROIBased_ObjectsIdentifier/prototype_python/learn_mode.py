#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


WINDOW_NAME = "Object Learning"
MEMORY_FILE = "object_memory.npz"


class MobileNetV3Embedding:
    """
    Uses the feature extractor of pretrained MobileNetV3-Small as an
    image embedding generator.

    Output:
        L2-normalized float32 embedding vector.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)

        self.model = model.features.to(self.device).eval()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device).eval()
        self.transform = weights.transforms()

        # Feature dimension for MobileNetV3-Small's final feature map.
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            feature_map = self.model(dummy)
            pooled = self.pool(feature_map)
            self.embedding_dim = int(pooled.flatten(1).shape[1])

        print(f"[MODEL] device={self.device}")
        print(f"[MODEL] embedding_dim={self.embedding_dim}")

    @torch.inference_mode()
    def extract(self, bgr_image: np.ndarray) -> np.ndarray:
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Invalid/empty image.")

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # torchvision transform expects a PIL image or tensor depending on version.
        # Using a tensor here keeps dependencies simple.
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        tensor = tensor.to(self.device)

        input_tensor = self.transform(tensor).unsqueeze(0)

        feature_map = self.model(input_tensor)
        pooled = self.pool(feature_map).flatten(1)

        # Cosine similarity works best with L2-normalized embeddings.
        embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return embedding[0].detach().cpu().numpy().astype(np.float32)


def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def get_name() -> str:
    while True:
        name = input("Object name (example: red_mug): ").strip()

        if not name:
            print("Object name cannot be empty.")
            continue

        # Keep names safe for display and future filesystem usage.
        name = " ".join(name.split())
        return name


def load_memory(path: Path):
    """
    Returns:
        names: list[str]
        vectors: float32 ndarray [N, D]
        offsets: int64 ndarray [num_objects + 1]
    """
    if not path.exists():
        return [], np.empty((0, 0), dtype=np.float32), np.array([0], dtype=np.int64)

    data = np.load(path, allow_pickle=True)

    names = [str(x) for x in data["names"].tolist()]
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    offsets = np.asarray(data["offsets"], dtype=np.int64)

    if vectors.ndim == 1 and vectors.size > 0:
        vectors = vectors.reshape(1, -1)

    return names, vectors, offsets


def save_memory(path: Path, names, vectors: np.ndarray, offsets: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        names=np.asarray(names, dtype=object),
        vectors=np.asarray(vectors, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        version=np.asarray([1], dtype=np.int32),
    )

    print(f"[MEMORY] Saved: {path}")
    print(f"[MEMORY] Objects: {len(names)}")
    print(f"[MEMORY] Vectors: {len(vectors)}")


def rebuild_offsets(counts):
    offsets = [0]
    total = 0

    for count in counts:
        total += int(count)
        offsets.append(total)

    return np.asarray(offsets, dtype=np.int64)


def add_object_to_memory(
    path: Path,
    object_name: str,
    new_vectors: np.ndarray,
    embedding_dim: int,
):
    names, old_vectors, old_offsets = load_memory(path)

    if len(old_vectors) > 0 and old_vectors.shape[1] != embedding_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: "
            f"memory={old_vectors.shape[1]}, model={embedding_dim}. "
            f"Delete {path} and learn again."
        )

    # Replace an existing object with the newly learned samples.
    object_vectors = {}

    for i, name in enumerate(names):
        start = int(old_offsets[i])
        end = int(old_offsets[i + 1])
        object_vectors[name] = old_vectors[start:end]

    object_vectors[object_name] = new_vectors

    new_names = list(object_vectors.keys())
    grouped = [object_vectors[name] for name in new_names]

    all_vectors = np.concatenate(grouped, axis=0) if grouped else np.empty(
        (0, embedding_dim), dtype=np.float32
    )
    counts = [len(v) for v in grouped]
    new_offsets = rebuild_offsets(counts)

    save_memory(path, new_names, all_vectors, new_offsets)


def draw_overlay(frame, roi, samples_taken, target_samples, object_name):
    display = frame.copy()

    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        display,
        f"Object: {object_name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"Samples: {samples_taken}/{target_samples}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        "SPACE = select ROI   ESC = quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return display


def main():
    parser = argparse.ArgumentParser(
        description="Learn an object into an embedding memory."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--interval", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--memory", type=str, default=MEMORY_FILE)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, etc.",
    )
    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    if args.interval < 0:
        raise ValueError("--interval must be >= 0")

    memory_path = Path(args.memory)

    print("=== Tiny Object Memory : LEARN MODE ===")
    print()
    print("Workflow:")
    print("  1. Camera opens")
    print("  2. Press SPACE")
    print("  3. Select the object with the mouse")
    print("  4. Press ENTER/SPACE in ROI window")
    print("  5. Slowly rotate the object")
    print("  6. Embeddings are saved to object_memory.npz")
    print()

    object_name = get_name()

    extractor = MobileNetV3Embedding(args.device)
    cap = open_camera(args.camera, args.width, args.height, args.fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    roi = None
    last_sample_time = 0.0
    embeddings = []

    print()
    print("[INFO] Press SPACE to select the object ROI.")
    print("[INFO] Press ESC to quit.")

    try:
        while len(embeddings) < args.samples:
            ok, frame = cap.read()

            if not ok:
                print("[ERROR] Failed to read camera frame.")
                break

            display = draw_overlay(
                frame,
                roi,
                len(embeddings),
                args.samples,
                object_name,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[INFO] Cancelled.")
                return

            if key == 32:  # SPACE
                if roi is None:
                    selected = cv2.selectROI(
                        WINDOW_NAME,
                        frame,
                        showCrosshair=True,
                        fromCenter=False,
                    )

                    x, y, w, h = map(int, selected)

                    if w <= 0 or h <= 0:
                        print("[INFO] Invalid ROI. Try again.")
                        continue

                    roi = (x, y, w, h)
                    last_sample_time = 0.0

                    print(
                        "[INFO] ROI selected. "
                        "Slowly rotate the object while keeping it in the ROI."
                    )
                    continue

                # Manual sample trigger after ROI is already selected.
                now = time.monotonic()
                if now - last_sample_time < args.interval:
                    continue

                x, y, w, h = roi
                crop = frame[y : y + h, x : x + w]

                if crop.size == 0:
                    print("[WARN] Empty ROI.")
                    continue

                try:
                    emb = extractor.extract(crop)
                    embeddings.append(emb)
                    last_sample_time = now

                    print(
                        f"[LEARN] sample={len(embeddings):02d}/"
                        f"{args.samples}  dim={len(emb)}"
                    )
                except Exception as exc:
                    print(f"[ERROR] Embedding extraction failed: {exc}")

        if len(embeddings) == 0:
            print("[INFO] No samples collected.")
            return

    finally:
        cap.release()
        cv2.destroyAllWindows()

    new_vectors = np.stack(embeddings).astype(np.float32)

    # Average one sample is NOT used as the only representation.
    # Every captured view is stored for robust matching across angles.
    add_object_to_memory(
        memory_path,
        object_name,
        new_vectors,
        extractor.embedding_dim,
    )

    print()
    print(f"[DONE] Learned object: {object_name}")
    print(f"[DONE] Samples: {len(new_vectors)}")
    print(f"[DONE] Memory file: {memory_path}")


if __name__ == "__main__":
    main()
