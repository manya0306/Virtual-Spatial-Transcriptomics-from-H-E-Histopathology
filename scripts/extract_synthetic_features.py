"""Build model-ready features for the SIMULATED 1,000-patch dataset.

Mirrors the real pipeline: frozen ImageNet ResNet18 (512-d, ``weights.transforms()``)
and 6-nearest-neighbour mean features computed WITHIN each sample.

Outputs (outputs/synthetic_features/): resnet18_features.npy, neighbor_features.npy,
metadata.csv, targets.csv, backend.txt.

Usage:
    python scripts/extract_synthetic_features.py --data-dir data/synthetic_spatial_transcriptomics_1000
    # offline smoke test only (NOT ResNet features, do not report results):
    python scripts/extract_synthetic_features.py --data-dir ... --backend handcrafted
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "synthetic_features"
N_NEIGHBORS = 6


def resnet_features(paths, batch=32):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    w = ResNet18_Weights.DEFAULT
    model = resnet18(weights=w)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = w.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            feats.append(model(x).numpy())
            print(f"  {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    return np.vstack(feats).astype(np.float32)


def handcrafted_features(paths):
    """Colour/density statistics + coarse 4x4 grid means. SMOKE TEST ONLY."""
    out = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
        grid = a.reshape(4, 64, 4, 64, 3).mean((1, 3)).ravel()
        out.append(np.concatenate([a.mean((0, 1)), a.std((0, 1)), [(a.mean(2) < 0.45).mean()], grid]))
    return np.array(out, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backend", choices=["resnet18", "handcrafted"], default="resnet18")
    a = ap.parse_args()
    d = Path(a.data_dir)
    meta = pd.read_csv(d / "metadata.csv")
    expr = pd.read_csv(d / "expression_56_genes.csv")
    if not (meta.patch_id.values == expr.patch_id.values).all():
        raise SystemExit("metadata / expression patch_id order mismatch")
    paths = [d / "images" / f"{p}.png" for p in meta.patch_id]
    feats = resnet_features(paths) if a.backend == "resnet18" else handcrafted_features(paths)

    neigh = np.zeros_like(feats)
    for s in meta.sample_id.unique():
        ix = np.where(meta.sample_id.values == s)[0]
        k = min(N_NEIGHBORS + 1, len(ix))
        _, nn = NearestNeighbors(n_neighbors=k).fit(meta.loc[ix, ["x", "y"]].values).kneighbors(
            meta.loc[ix, ["x", "y"]].values)
        neigh[ix] = feats[ix][nn[:, 1:]].mean(1)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "resnet18_features.npy", feats)
    np.save(OUT / "neighbor_features.npy", neigh)
    pd.DataFrame({"sample": meta.sample_id, "barcode": meta.patch_id, "x_coordinate": meta.x,
                  "y_coordinate": meta.y, "tissue_type": meta.tissue_type, "split": meta.split}
                 ).to_csv(OUT / "metadata.csv", index=False)
    expr.to_csv(OUT / "targets.csv", index=False)
    (OUT / "backend.txt").write_text(a.backend + "\n")
    print(f"\nSaved {feats.shape} features (backend={a.backend}) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
