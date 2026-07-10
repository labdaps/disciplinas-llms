"""
LAB-53: Grafo de similaridade por cosseno entre disciplinas.

Saida:
  data/graph_<fonte>.json  — nos e arestas para o Streamlit

Uso:
  python scripts/build_graph.py [--fonte saude_publica] [--threshold 0.15]
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", default="saude_publica")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--n-clusters", type=int, default=10)
    args = parser.parse_args()

    print(f"Fonte: {args.fonte}, threshold: {args.threshold}")

    X = load_npz(str(DATA / f"embeddings_{args.fonte}.npz"))
    meta = json.loads((DATA / f"meta_{args.fonte}.json").read_text("utf-8"))
    disc = json.loads((DATA / f"{args.fonte}.json").read_text("utf-8"))

    print(f"Calculando matriz de similaridade {X.shape[0]}x{X.shape[0]}...")
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    n = len(meta)

    # Clustering por KMeans sobre a matriz densa de similaridade
    n_clusters = args.n_clusters
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(sim)
    print(f"Clusters KMeans (k={n_clusters}): {dict(zip(*np.unique(clusters, return_counts=True)))}")

    # Construir grafo NetworkX com arestas acima do threshold
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= args.threshold:
                G.add_edge(i, j, weight=s)

    print(f"Arestas (sim >= {args.threshold}): {G.number_of_edges()}")

    # Layout force-directed (spring layout ponderado)
    print("Calculando layout spring...")
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.5 / (n ** 0.5), iterations=60)

    # Montar JSON
    nos_json = [
        {
            "id": i,
            "codigo": meta[i]["Codigo"],
            "nome": meta[i]["Nome"],
            "unidade": meta[i]["Unidade"],
            "departamento": meta[i]["Departamento"],
            "cluster": int(clusters[i]),
            "grau": G.degree(i),
            "x": round(float(pos[i][0]), 4),
            "y": round(float(pos[i][1]), 4),
        }
        for i in range(n)
    ]

    arestas_json = [
        {"source": u, "target": v, "sim": round(d["weight"], 4)}
        for u, v, d in G.edges(data=True)
    ]

    out = {
        "threshold": args.threshold,
        "n_nos": n,
        "n_arestas": G.number_of_edges(),
        "n_clusters": n_clusters,
        "nos": nos_json,
        "arestas": arestas_json,
    }

    out_path = DATA / f"graph_{args.fonte}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    print(f"Grafo salvo: {out_path}")

    graus = [G.degree(i) for i in range(n)]
    print(f"Grau medio: {np.mean(graus):.1f}, max: {max(graus)}, isolados: {sum(1 for g in graus if g == 0)}")


if __name__ == "__main__":
    main()
