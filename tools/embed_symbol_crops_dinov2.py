from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_images(rows: list[dict[str, Any]]) -> list[Image.Image]:
    images = []
    for row in rows:
        images.append(Image.open(row["crop_path"]).convert("RGB"))
    return images


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_embeddings(
    rows: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    device_name: str,
    local_files_only: bool,
) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = torch.device(device_name)
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            images = load_images(batch_rows)
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if getattr(outputs, "pooler_output", None) is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features, dim=1)
            embeddings.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)


def write_nearest_neighbors(rows: list[dict[str, Any]], embeddings: np.ndarray, out_path: Path, top_k: int) -> None:
    if embeddings.shape[0] == 0:
        write_json(out_path, [])
        return
    from sklearn.neighbors import NearestNeighbors

    n_neighbors = min(max(1, top_k + 1), embeddings.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    payload = []
    for query_idx, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        neighbors = []
        for distance, neighbor_idx in zip(row_distances, row_indices):
            if int(neighbor_idx) == query_idx:
                continue
            neighbors.append(
                {
                    "index": int(neighbor_idx),
                    "crop_path": rows[int(neighbor_idx)]["crop_path"],
                    "class": rows[int(neighbor_idx)]["class"],
                    "cosine_distance": float(distance),
                }
            )
            if len(neighbors) >= top_k:
                break
        payload.append(
            {
                "index": query_idx,
                "crop_path": rows[query_idx]["crop_path"],
                "class": rows[query_idx]["class"],
                "neighbors": neighbors,
            }
        )
    write_json(out_path, payload)


def write_clusters(rows: list[dict[str, Any]], embeddings: np.ndarray, out_path: Path, cluster_k: int) -> None:
    if cluster_k <= 0 or embeddings.shape[0] == 0:
        return
    from sklearn.cluster import KMeans

    k = min(cluster_k, embeddings.shape[0])
    labels = KMeans(n_clusters=k, n_init="auto", random_state=17).fit_predict(embeddings)
    clusters: dict[str, list[dict[str, Any]]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(str(int(label)), []).append(
            {
                "index": idx,
                "crop_path": rows[idx]["crop_path"],
                "class": rows[idx]["class"],
            }
        )
    write_json(out_path, {"cluster_k": k, "clusters": clusters})


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed symbol crops with DINOv2 for clustering and ReID-style QA.")
    parser.add_argument("--metadata", type=Path, default=Path("outputs/v2_symbol_crops/metadata.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/v2_dinov2_embeddings"))
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cluster-k", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata without loading model weights.")
    args = parser.parse_args()

    rows = read_jsonl(args.metadata, limit=args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        payload = {
            "metadata": str(args.metadata),
            "rows": len(rows),
            "model": args.model,
            "first_rows": rows[:3],
        }
        write_json(args.out_dir / "dry_run_summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    device = choose_device(args.device)
    embeddings = compute_embeddings(rows, args.model, args.batch_size, device, args.local_files_only)
    np.save(args.out_dir / "embeddings.npy", embeddings)
    metadata_copy = args.out_dir / "metadata.jsonl"
    metadata_copy.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_nearest_neighbors(rows, embeddings, args.out_dir / "nearest_neighbors.json", args.top_k)
    write_clusters(rows, embeddings, args.out_dir / "clusters.json", args.cluster_k)
    summary = {
        "metadata": str(metadata_copy),
        "embeddings": str(args.out_dir / "embeddings.npy"),
        "rows": len(rows),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0,
        "model": args.model,
        "device": device,
        "nearest_neighbors": str(args.out_dir / "nearest_neighbors.json"),
        "clusters": str(args.out_dir / "clusters.json") if args.cluster_k > 0 else None,
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
