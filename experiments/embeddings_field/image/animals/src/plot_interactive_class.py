from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.decomposition import PCA


def load_manifest(manifest_path: Path) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_base64_image(path: str, size=(400, 400)) -> str:
    try:
        img = Image.open(path)
        img.thumbnail(size)
        buffered = BytesIO()
        # Convert to RGB if necessary for JPEG
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffered, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error encoding {path}: {e}")
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive plot with image preview on click.")
    parser.add_argument("--class-name", default="bear", help="The animal class to highlight.")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments/embeddings_field/image/animals"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    # Paths
    embeddings_path = args.experiment_root / "artifacts" / "embeddings" / "full_raw.npy"
    labels_path = args.experiment_root / "data" / "processed" / "full" / "labels.npy"
    manifest_path = args.experiment_root / "data" / "processed" / "full" / "manifest.jsonl"
    metadata_path = args.experiment_root / "data" / "processed" / "metadata.json"

    if args.output_root:
        output_root = args.output_root
    else:
        output_root = args.experiment_root / "reports" / "plots" / "interactive"
    output_root.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    manifest = load_manifest(manifest_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    label_to_id = metadata["label_to_id"]
    if args.class_name not in label_to_id:
        print(f"Error: Class '{args.class_name}' not found.")
        return

    target_id = label_to_id[args.class_name]
    
    # PCA
    print("Computing PCA...")
    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)

    mask = labels == target_id
    target_coords = coords[mask]
    target_manifest = [row for i, row in enumerate(manifest) if mask[i]]

    print(f"Encoding {len(target_coords)} images to Base64 (this might take a few seconds)...")
    base64_images = []
    for row in target_manifest:
        b64 = get_base64_image(row["source_path"])
        if not b64:
            print(f"Warning: Failed to encode {row['source_path']}")
        base64_images.append(b64)

    data = {
        "x": target_coords[:, 0],
        "y": target_coords[:, 1],
        "filename": [row["filename"] for row in target_manifest],
        "path": [row["source_path"] for row in target_manifest],
        "base64": base64_images
    }

    fig = px.scatter(
        data,
        x="x",
        y="y",
        hover_data=["filename"],
        custom_data=["base64", "filename", "path"],
        title=f"Interactive Projection: {args.class_name.capitalize()} (Click dots to see images)",
        labels={"x": "PCA 1", "y": "PCA 2"},
        template="plotly_white"
    )

    fig.update_traces(
        marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate="<b>%{customdata[1]}</b><br>PCA1: %{x:.3f}<br>PCA2: %{y:.3f}<extra></extra>"
    )

    # Generate HTML with custom JS for click handling
    # Use include_plotlyjs='cdn' to keep file size smaller
    config = {'responsive': True}
    plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config=config)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{args.class_name.capitalize()} Interactive Explorer</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; display: flex; flex-direction: row; height: 100vh; width: 100vw; overflow: hidden; }}
            #plot-container {{ flex: 1; height: 100%; position: relative; overflow: hidden; background: #fff; }}
            #side-panel {{ 
                width: 450px; 
                min-width: 450px; 
                flex-shrink: 0; 
                border-left: 2px solid #eee; 
                padding: 20px; 
                background: #fdfdfd; 
                overflow-y: auto; 
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                box-shadow: -2px 0 10px rgba(0,0,0,0.05);
                z-index: 100;
                box-sizing: border-box;
            }}
            #image-display {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-top: 20px; border-radius: 4px; }}
            #metadata {{ margin-top: 20px; width: 100%; font-size: 14px; word-break: break-all; border-top: 1px solid #eee; padding-top: 15px; line-height: 1.4; }}
            .placeholder {{ color: #999; margin-top: 120px; text-align: center; font-style: italic; }}
            h3 {{ margin-top: 0; color: #222; border-bottom: 2px solid #3498db; padding-bottom: 8px; width: 100%; text-align: center; }}
            .plotly-graph-div {{ height: 100% !important; width: 100% !important; }}
        </style>
    </head>
    <body>
        <div id="plot-container">
            {plotly_html}
        </div>
        <div id="side-panel">
            <h3>Image Preview</h3>
            <div id="content">
                <p class="placeholder">Click a dot on the scatter plot<br>to view the animal image.</p>
            </div>
            <img id="image-display" style="display:none;" alt="Preview">
            <div id="metadata"></div>
        </div>

        <script>
            function initPlot() {{
                var plotDivs = document.getElementsByClassName('plotly-graph-div');
                if (plotDivs.length > 0) {{
                    var plotDiv = plotDivs[0];
                    console.log("Plotly div found, attaching click handler...");

                    plotDiv.on('plotly_click', function(data) {{
                        console.log("Point clicked!", data);
                        var point = data.points[0];

                        if (point.customdata && point.customdata[0]) {{
                            var base64 = point.customdata[0];
                            var filename = point.customdata[1];
                            var path = point.customdata[2];

                            document.getElementById('content').style.display = 'none';
                            var img = document.getElementById('image-display');
                            img.src = base64;
                            img.style.display = 'block';

                            document.getElementById('metadata').innerHTML = 
                                "<strong>Filename:</strong> " + filename + "<br><br>" +
                                "<strong>Path:</strong> <span style='color: #666; font-family: monospace; font-size: 12px;'>" + path + "</span>";
                        }} else {{
                            console.error("No image data found for this point!");
                            alert("Error: Could not load image data for this point.");
                        }}
                    }});
                    
                    // Force resize to ensure it fits container
                    window.dispatchEvent(new Event('resize'));
                }} else {{
                    setTimeout(initPlot, 100);
                }}
            }}
            initPlot();
        </script>
    </body>
    </html>
    """

    output_path = output_root / f"{args.class_name}_interactive_preview.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"Interactive plot with preview saved to: {output_path}")


if __name__ == "__main__":
    main()
