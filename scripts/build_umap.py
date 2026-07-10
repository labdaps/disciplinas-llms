"""
LAB-54: Reducao dimensional UMAP + clusterizacao HDBSCAN.

Saida:
  data/umap_<fonte>.json  — coordenadas 2D e labels de cluster para o Streamlit

Uso:
  python scripts/build_umap.py [--fonte saude_publica]
"""

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import umap
from scipy.sparse import load_npz

DATA = Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", default="saude_publica")
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    args = parser.parse_args()

    print(f"Fonte: {args.fonte}")
    X = load_npz(str(DATA / f"embeddings_{args.fonte}.npz"))
    meta = json.loads((DATA / f"meta_{args.fonte}.json").read_text("utf-8"))
    disc = json.loads((DATA / f"{args.fonte}.json").read_text("utf-8"))

    print(f"Matriz: {X.shape}")

    # UMAP
    print(f"UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(X)
    print(f"Coords shape: {coords.shape}")

    # HDBSCAN sobre o espaco UMAP
    print(f"HDBSCAN (min_cluster_size={args.min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = int((labels == -1).sum())
    print(f"Clusters: {n_clusters}, ruido (label=-1): {n_ruido}/{len(labels)}")

    # Montar JSON
    pontos = []
    for i, d in enumerate(disc):
        pontos.append({
            "id": i,
            "codigo": meta[i]["Codigo"],
            "nome": meta[i]["Nome"],
            "unidade": meta[i]["Unidade"],
            "departamento": meta[i]["Departamento"],
            "cluster": int(labels[i]),
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "ementa": d.get("Ementa", "")[:200],
        })

    out = {
        "n_pontos": len(pontos),
        "n_clusters": n_clusters,
        "n_ruido": n_ruido,
        "params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "min_cluster_size": args.min_cluster_size,
        },
        "pontos": pontos,
    }

    out_path = DATA / f"umap_{args.fonte}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print(f"Salvo: {out_path}")


if __name__ == "__main__":
    main()
