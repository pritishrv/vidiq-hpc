from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_manifest(manifest_path: Path) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_base64_image(path: str, size=(200, 200)) -> str:
    try:
        img = Image.open(path)
        img.thumbnail(size)
        buffered = BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffered, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        return ""


def analyze_class_clusters(class_name: str, embeddings: np.ndarray, labels: np.ndarray, manifest: list[dict], label_to_id: dict) -> str:
    target_id = label_to_id[class_name]
    mask = labels == target_id
    class_embs = embeddings[mask]
    class_manifest = [row for i, row in enumerate(manifest) if mask[i]]

    # K-Means to find the two clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(class_embs)
    
    # PCA for visualization context (optional, but good to know)
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(class_embs)

    html_parts = [f"<h2>Analysis for: {class_name.capitalize()}</h2>", "<div class='comparison-container'>"]
    
    for cluster_id in [0, 1]:
        html_parts.append(f"<div class='cluster-col'><h3>Cluster {cluster_id}</h3><div class='grid'>")
        
        # Get indices for this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        # Pick 12 representative images (closest to centroid of this cluster)
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(class_embs[cluster_indices] - cluster_center, axis=1)
        closest_indices = cluster_indices[np.argsort(distances)[:12]]
        
        for idx in closest_indices:
            row = class_manifest[idx]
            b64 = get_base64_image(row["source_path"])
            html_parts.append(f"<div class='img-card'><img src='{b64}'><br><span class='filename'>{row['filename']}</span></div>")
        
        html_parts.append("</div></div>")
    
    html_parts.append("</div>")
    return "\n".join(html_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/embeddings_field/image/animals"))
    args = parser.parse_args()

    embeddings = np.load(args.experiment_root / "artifacts" / "embeddings" / "full_raw.npy")
    labels = np.load(args.experiment_root / "data" / "processed" / "full" / "labels.npy")
    manifest = load_manifest(args.experiment_root / "data" / "processed" / "full" / "manifest.jsonl")
    with open(args.experiment_root / "data" / "processed" / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    label_to_id = metadata["label_to_id"]
    
    classes_to_analyze = ["bear", "dog", "cat", "panda"]
    
    report_content = []
    for cls in classes_to_analyze:
        print(f"Analyzing {cls}...")
        report_content.append(analyze_class_clusters(cls, embeddings, labels, manifest, label_to_id))

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Animal Cluster Comparison Analysis</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background: #f0f2f5; }}
            .comparison-container {{ display: flex; gap: 20px; margin-bottom: 50px; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .cluster-col {{ flex: 1; }}
            .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
            .img-card {{ text-align: center; background: #eee; padding: 5px; border-radius: 4px; }}
            .img-card img {{ width: 100%; height: auto; border-radius: 2px; }}
            .filename {{ font-size: 10px; color: #666; word-break: break-all; }}
            h2 {{ border-bottom: 2px solid #333; padding-bottom: 5px; margin-top: 40px; }}
            h3 {{ color: #555; text-align: center; background: #f8f8f8; padding: 5px; }}
        </style>
    </head>
    <body>
        <h1>CLIP Embedding Cluster Analysis</h1>
        <p>This report compares the two distinct clusters found in the PCA projections for various animal classes.</p>
        {"".join(report_content)}
    </body>
    </html>
    """

    output_path = args.experiment_root / "reports" / "cluster_comparison_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
