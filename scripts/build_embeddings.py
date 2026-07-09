"""
LAB-52: Gera embeddings TF-IDF a partir de objetivos, ementa e bibliografia.

Saida:
  data/embeddings.npz   — matriz esparsa TF-IDF
  data/meta.json        — lista de dicts {indice, Codigo, Nome, Unidade, Departamento}
  data/tfidf_model.pkl  — vectorizer serializado (para reusar na busca do app)

Uso:
  python scripts/build_embeddings.py [--fonte saude_publica|disciplinas|all]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).parent.parent / "data"


def carregar(fonte: str) -> list[dict]:
    if fonte == "saude_publica":
        return json.loads((DATA / "saude_publica.json").read_text("utf-8"))
    if fonte == "disciplinas":
        return json.loads((DATA / "disciplinas.json").read_text("utf-8"))
    # all: concatena os dois, normalizando nomes de campos
    sp = json.loads((DATA / "saude_publica.json").read_text("utf-8"))
    en = json.loads((DATA / "disciplinas.json").read_text("utf-8"))
    # normaliza campos do dataset em ingles
    for d in en:
        d.setdefault("Ementa", d.get("Objectives", ""))
        d.setdefault("Objetivos", d.get("Rationale", ""))
        d.setdefault("Bibliografia", d.get("Bibliography", ""))
        d.setdefault("Codigo", d.get("Code", ""))
        d.setdefault("Nome", d.get("Name", ""))
        d.setdefault("Unidade", d.get("School name") or d.get("School", ""))
        d.setdefault("Departamento", "")
    return sp + en


def montar_corpus(disciplinas: list[dict]) -> list[str]:
    textos = []
    for d in disciplinas:
        partes = [
            d.get("Ementa", ""),
            d.get("Objetivos", ""),
            d.get("Programa", ""),
            d.get("Bibliografia", ""),
        ]
        textos.append(" ".join(p for p in partes if p).strip())
    return textos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", default="saude_publica",
                        choices=["saude_publica", "disciplinas", "all"])
    args = parser.parse_args()

    print(f"Fonte: {args.fonte}")
    disciplinas = carregar(args.fonte)
    print(f"Disciplinas carregadas: {len(disciplinas)}")

    corpus = montar_corpus(disciplinas)
    nao_vazios = sum(1 for t in corpus if t)
    print(f"Textos nao vazios: {nao_vazios}/{len(corpus)}")

    vec = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X = vec.fit_transform(corpus)
    print(f"Matriz TF-IDF: {X.shape} ({X.nnz} nao-zeros)")

    # Salvar matriz
    out_npz = DATA / f"embeddings_{args.fonte}.npz"
    save_npz(str(out_npz), X)
    print(f"Matriz salva: {out_npz}")

    # Salvar modelo
    out_pkl = DATA / f"tfidf_{args.fonte}.pkl"
    out_pkl.write_bytes(pickle.dumps(vec))
    print(f"Modelo salvo: {out_pkl}")

    # Salvar metadados
    meta = [
        {
            "indice": i,
            "Codigo": d.get("Codigo", ""),
            "Nome": d.get("Nome", ""),
            "Unidade": d.get("Unidade", ""),
            "Departamento": d.get("Departamento", ""),
            "Creditos_aula": d.get("Creditos_aula", ""),
            "Carga_horaria": d.get("Carga_horaria", ""),
            "texto_len": len(corpus[i]),
        }
        for i, d in enumerate(disciplinas)
    ]
    out_meta = DATA / f"meta_{args.fonte}.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(f"Meta salvo: {out_meta}")

    # Top termos por disciplina (amostra)
    feature_names = vec.get_feature_names_out()
    print("\n--- Top 5 termos por disciplina (amostra 5) ---")
    for i in range(min(5, len(disciplinas))):
        row = X[i].toarray()[0]
        top5 = feature_names[row.argsort()[-5:][::-1]]
        print(f"  [{meta[i]['Codigo']}] {meta[i]['Nome'][:50]}")
        print(f"    {list(top5)}")


if __name__ == "__main__":
    main()
