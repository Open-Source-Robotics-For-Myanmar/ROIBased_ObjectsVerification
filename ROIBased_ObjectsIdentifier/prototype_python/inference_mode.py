#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


WINDOW_NAME = "Object Inference"
MEMORY_FILE = "object_memory.npz"


class MobileNetV3Embedding:
    """
    Same embedding pipeline as learn_mode.py.
    The exact same preprocessing/model must be used during inference.
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
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        tensor = tensor.to(self.device)

        input_tensor = self.transform(tensor).unsqueeze(0)

        feature_map = self.model(input_tensor)
        pooled = self.pool(feature_map).flatten(1)
        embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return embedding[0].detach().cpu().numpy().astype(np.float32)


def load_memory(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Memory file not found: {path}\n"
            f"Run learn_mode.py first."
        )

    data = np.load(path, allow_pickle=True)

    names = [str(x) for x in data["names"].tolist()]
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    offsets = np.asarray(data["offsets"], dtype=np.int64)

    if vectors.ndim == 1 and vectors.size > 0:
        vectors = vectors.reshape(1, -1)

    if len(offsets) != len(names) + 1:
        raise RuntimeError("Corrupt memory file: offsets/names mismatch.")

    return names, vectors, offsets


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Because all vectors are already L2-normalized,
    cosine similarity is simply a dot product.
    """
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        raise ValueError("Query embedding has zero norm.")

    m_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_matrix = matrix / np.maximum(m_norm, 1e-12)

    return safe_matrix @ (query / q_norm)


def match_object(
    query_vector: np.ndarray,
    names,
    vectors: np.ndarray,
    offsets: np.ndarray,
    top_k: int = 5,
):
    """
    For each stored object:
        1. compare query against all stored views
        2. take top-k view similarities
        3. average those top-k scores

    This is more robust than matching against only one saved view.
    """
    object_results = []

    for i, name in enumerate(names):
        start = int(offsets[i])
        end = int(offsets[i + 1])

        object_vectors = vectors[start:end]

        if len(object_vectors) == 0:
            continue

        scores = cosine_similarity(query_vector, object_vectors)
        order = np.argsort(scores)[::-1]

        k = min(top_k, len(scores))
        top_scores = scores[order[:k]]

        aggregate_score = float(np.mean(top_scores))
        best_score = float(top_scores[0])
        best_view = int(order[0])

        object_results.append(
            {
                "name": name,
                "score": aggregate_score,
                "best_score": best_score,
                "best_view": best_view,
                "num_views": len(scores),
                "top_scores": top_scores,
            }
        )

    if not object_results:
        return None, []

    object_results.sort(key=lambda x: x["score"], reverse=True)

    return object_results[0], object_results


def open_camera(camera_index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def draw_result(
    frame,
    roi,
    result,
    threshold,
    status_message,
):
    display = frame.copy()

    if roi is not None:
        x, y, w, h = roi

        if result is not None and result["score"] >= threshold:
            box_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            box_color,
            2,
        )

    if result is None:
        title = "NO MEMORY"
        score_text = ""
    else:
        matched = result["score"] >= threshold

        title = (
            f"MATCH: {result['name']}"
            if matched
            else "NO MATCH: UNKNOWN"
        )

        score_text = (
            f"score={result['score']:.3f}  "
            f"best={result['best_score']:.3f}  "
            f"view={result['best_view']}"
        )

    cv2.putText(
        display,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if score_text:
        cv2.putText(
            display,
            score_text,
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        display,
        f"threshold={threshold:.3f}",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if status_message:
        cv2.putText(
            display,
            status_message,
            (20, display.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        display,
        "SPACE = select ROI / recognize    ESC = quit",
        (20, display.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return display


def main():
    parser = argparse.ArgumentParser(
        description="Recognize an object from an embedding memory."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--memory", type=str, default=MEMORY_FILE)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    memory_path = Path(args.memory)

    print("=== Tiny Object Memory : INFERENCE MODE ===")

    names, vectors, offsets = load_memory(memory_path)

    if len(names) == 0:
        raise RuntimeError(
            f"No objects stored in {memory_path}. "
            f"Run learn_mode.py first."
        )

    print()
    print(f"[MEMORY] File: {memory_path}")
    print(f"[MEMORY] Objects: {len(names)}")
    print(f"[MEMORY] Total views: {len(vectors)}")
    print(f"[MEMORY] Names: {', '.join(names)}")

    extractor = MobileNetV3Embedding(args.device)

    if vectors.shape[1] != extractor.embedding_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: "
            f"memory={vectors.shape[1]}, model={extractor.embedding_dim}"
        )

    cap = open_camera(args.camera, args.width, args.height, args.fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    roi = None
    current_result = None
    status_message = "Press SPACE to select ROI"

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[ERROR] Failed to read camera frame.")
                break

            display = draw_result(
                frame,
                roi,
                current_result,
                args.threshold,
                status_message,
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key == 32:  # SPACE
                selected = cv2.selectROI(
                    WINDOW_NAME,
                    frame,
                    showCrosshair=True,
                    fromCenter=False,
                )

                x, y, w, h = map(int, selected)

                if w <= 0 or h <= 0:
                    status_message = "Invalid ROI"
                    current_result = None
                    continue

                roi = (x, y, w, h)

                crop = frame[y : y + h, x : x + w]

                try:
                    print("[INFERENCE] Extracting embedding...")
                    query_vector = extractor.extract(crop)

                    best, all_results = match_object(
                        query_vector,
                        names,
                        vectors,
                        offsets,
                        top_k=args.top_k,
                    )

                    current_result = best

                    if best is None:
                        status_message = "No valid objects in memory"
                        continue

                    matched = best["score"] >= args.threshold

                    if matched:
                        status_message = (
                            f"MATCH -> {best['name']} "
                            f"(score={best['score']:.3f})"
                        )
                    else:
                        status_message = (
                            f"UNKNOWN "
                            f"(best={best['name']}, "
                            f"score={best['score']:.3f})"
                        )

                    print()
                    print("=== RESULTS ===")

                    for rank, result in enumerate(all_results, start=1):
                        print(
                            f"{rank:02d}. "
                            f"{result['name']:<20} "
                            f"score={result['score']:.4f} "
                            f"best={result['best_score']:.4f} "
                            f"views={result['num_views']}"
                        )

                    print(
                        f"\nFINAL: "
                        f"{'MATCH' if matched else 'NO MATCH'}"
                        f" -> {best['name'] if matched else 'UNKNOWN'}"
                    )

                except Exception as exc:
                    print(f"[ERROR] Inference failed: {exc}")
                    current_result = None
                    status_message = "Inference error"

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
