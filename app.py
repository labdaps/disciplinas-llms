"""
Painel de Sobreposição de Disciplinas USP
Hub: busca por tópico via chat. Exploração analítica em card separado.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Disciplinas USP | LABDAPS",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── estilos ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* esconde sidebar e todos os controles relacionados */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"],
.st-emotion-cache-czk5ss,
#MainMenu { display: none !important; }

/* remove margem lateral que o Streamlit reserva para a sidebar */
.main .block-container { max-width: 780px; padding: 0 2rem 4rem; }

/* hero */
.hero { text-align: center; padding: 52px 0 24px; }
.hero-title { font-size: 2rem; font-weight: 700; color: #0f1730; margin-bottom: 8px; line-height: 1.2; }
.hero-sub { font-size: .95rem; color: #6b7280; margin-bottom: 28px; }

/* chips — nowrap para nao quebrar palavras */
.chip-wrap { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 24px; }
.chip-btn {
    background: #eef1fb; color: #223886;
    border: 1px solid #c7d0f0; border-radius: 20px;
    padding: 5px 14px; font-size: .82rem; font-weight: 500;
    white-space: nowrap; cursor: pointer;
    display: inline-block;
}

/* resultado */
.result-header {
    font-size: .72rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #6b7280; margin: 24px 0 12px;
}
.disc-card {
    background: #fff; border: 1px solid #e5e7eb;
    border-left: 4px solid #223886; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 10px;
}
.disc-card-dim { border-left-color: #d1d5db; }
.disc-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.disc-name { font-weight: 600; color: #0f1730; font-size: .95rem; }
.disc-meta { font-size: .76rem; color: #6b7280; margin-top: 3px; }
.sim-badge {
    font-size: .75rem; font-weight: 700; white-space: nowrap;
    padding: 3px 10px; border-radius: 12px; flex-shrink: 0;
}
.sim-high { background: #dcfce7; color: #166534; }
.sim-mid  { background: #fef9c3; color: #854d0e; }
.sim-low  { background: #f3f4f6; color: #6b7280; }
.disc-obj { font-size: .82rem; color: #374151; margin-top: 8px; line-height: 1.5; }

/* separador de cluster */
.cluster-sep {
    font-size: .7rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #223886;
    background: #eef1fb; border-radius: 6px;
    padding: 4px 12px; display: inline-block;
    margin: 18px 0 10px;
}

/* card exploração */
.explore-card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 22px 24px; margin-top: 40px;
}
.explore-title { font-size: 1rem; font-weight: 600; color: #0f1730; margin-bottom: 4px; }
.explore-sub { font-size: .83rem; color: #6b7280; margin-bottom: 0; }

/* kpis */
.kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }
.kpi { background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:12px 18px; flex:1; min-width:90px; text-align:center; }
.kpi-val { font-size: 1.5rem; font-weight: 700; color: #223886; }
.kpi-lbl { font-size: .68rem; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; margin-top: 2px; }

.section-tag {
    display: inline-block; background: #eef1fb; color: #223886;
    font-size: .7rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; padding: 3px 10px; border-radius: 20px; margin-bottom: 6px;
}

/* botoes de chip — container wrap, colunas auto-tamanho */
div[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 8px !important;
    justify-content: center !important;
}
div[data-testid="stHorizontalBlock"] > div {
    flex: 0 0 auto !important;
    min-width: fit-content !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"] button {
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    min-width: fit-content !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ── dados ─────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "disciplinas.json"
GRAPH_PATH = Path(__file__).parent / "data" / "graph_saude_publica.json"


@st.cache_data
def carregar_grafo():
    if not GRAPH_PATH.exists():
        return None
    return json.loads(GRAPH_PATH.read_text("utf-8"))

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["School"].str.match(r"^[A-Z]{2,8}$", na=False)].copy()
    df["credits_num"] = pd.to_numeric(df["Number of credits"], errors="coerce")
    df["texto"] = (
        df["Name"].fillna("") + " " +
        df["Objectives"].fillna("") + " " +
        df["Content"].fillna("")
    )
    return df.reset_index(drop=True)

@st.cache_resource
def build_index(df: pd.DataFrame):
    vec = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(df["texto"])
    return vec, normalize(X)

@st.cache_data
def build_analytics(_df: pd.DataFrame, n_clusters: int = 8):
    vec, X_norm = build_index(_df)
    sim = cosine_similarity(X_norm)
    coords = TruncatedSVD(n_components=2, random_state=42).fit_transform(X_norm)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_norm)
    feat = vec.get_feature_names_out()
    dense = X_norm.toarray()
    cluster_names = {
        c: " / ".join(feat[i] for i in dense[labels == c].mean(axis=0).argsort()[-3:][::-1])
        for c in range(n_clusters)
    }
    return sim, coords, labels, cluster_names

def buscar(query: str, df, top_k=40, threshold=0.008):
    vec, X_norm = build_index(df)
    q = normalize(vec.transform([query]))
    sims = cosine_similarity(q, X_norm)[0]
    ordem = np.argsort(sims)[::-1]
    return [{"idx": int(i), "sim": float(sims[i])} for i in ordem[:top_k] if sims[i] >= threshold]

df = carregar_dados()

# ── estado ────────────────────────────────────────────────────────────────────

if "historico" not in st.session_state:
    st.session_state.historico = []
if "query_atual" not in st.session_state:
    st.session_state.query_atual = ""
if "modo_explorar" not in st.session_state:
    st.session_state.modo_explorar = False

# ── hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-title">🎓 Explore disciplinas USP por tópico</div>
  <div class="hero-sub">Digite um tema e veja todas as disciplinas que se sobrepõem — de IA a epidemiologia.</div>
</div>
""", unsafe_allow_html=True)

# chips — labels curtos, sem use_container_width para nao esticar
exemplos = [
    "Machine Learning", "Epidemiologia", "Bioinformatica",
    "Estatistica", "Genomica", "Clinica",
]

cols = st.columns(len(exemplos))
for i, ex in enumerate(exemplos):
    with cols[i]:
        if st.button(ex, key=f"chip_{ex}"):
            st.session_state.query_atual = ex
            if ex not in st.session_state.historico:
                st.session_state.historico.append(ex)
            st.rerun()

# chat input
query_input = st.chat_input("Ex: machine learning, redes neurais, deep learning...")
if query_input:
    st.session_state.query_atual = query_input
    if query_input not in st.session_state.historico:
        st.session_state.historico.append(query_input)
    st.rerun()

# histórico compacto abaixo do input
if st.session_state.historico and len(st.session_state.historico) > 0:
    with st.expander(f"Buscas anteriores ({len(st.session_state.historico)})", expanded=False):
        for q in reversed(st.session_state.historico[-10:]):
            if st.button(q, key=f"hist_{q}", use_container_width=True):
                st.session_state.query_atual = q
                st.rerun()
        if st.button("Limpar", use_container_width=True):
            st.session_state.historico = []
            st.session_state.query_atual = ""
            st.rerun()

# ── resultados ────────────────────────────────────────────────────────────────

query = st.session_state.query_atual

if query:
    resultados = buscar(query, df)
    _, _, labels_all, cluster_names = build_analytics(df, 8)

    if not resultados:
        st.info("Nenhuma disciplina encontrada. Tente termos em inglês.")
    else:
        sim_max = max(r["sim"] for r in resultados)
        st.markdown(
            f'<div class="result-header">🔍 "{query}" — {len(resultados)} disciplinas com sobreposição</div>',
            unsafe_allow_html=True,
        )

        # agrupa por cluster
        grupos: dict[int, list] = {}
        for r in resultados:
            grupos.setdefault(int(labels_all[r["idx"]]), []).append(r)

        for cluster_id, itens in sorted(grupos.items(), key=lambda kv: max(r["sim"] for r in kv[1]), reverse=True):
            nome_cluster = cluster_names.get(cluster_id, str(cluster_id))
            st.markdown(
                f'<div class="cluster-sep">📂 {nome_cluster} &nbsp;({len(itens)} disciplina{"s" if len(itens) > 1 else ""})</div>',
                unsafe_allow_html=True,
            )
            for r in sorted(itens, key=lambda x: x["sim"], reverse=True):
                row = df.iloc[r["idx"]]
                pct = int(r["sim"] * 100)
                rel = r["sim"] / sim_max
                badge = "sim-high" if rel >= 0.6 else ("sim-mid" if rel >= 0.3 else "sim-low")
                card_cls = "disc-card" if rel >= 0.15 else "disc-card disc-card-dim"
                school = (row.get("School name") or row.get("School") or "").strip()
                credits = int(row["credits_num"]) if pd.notna(row["credits_num"]) else 0
                obj = (row.get("Objectives") or "").strip()[:220]
                if obj:
                    obj_html = f"<div class='disc-obj'>{obj}{'...' if len((row.get('Objectives') or '')) > 220 else ''}</div>"
                else:
                    obj_html = ""
                st.markdown(f"""
<div class="{card_cls}">
  <div class="disc-top">
    <div>
      <div class="disc-name">{row['Name']}</div>
      <div class="disc-meta">{school} &nbsp;|&nbsp; {credits} créditos</div>
    </div>
    <span class="sim-badge {badge}">{pct}% sobreposição</span>
  </div>
  {obj_html}
</div>""", unsafe_allow_html=True)

# ── card exploração analítica ─────────────────────────────────────────────────

st.markdown("""
<div class="explore-card">
  <div class="explore-title">📊 Exploração analítica</div>
  <div class="explore-sub">Mapa temático completo, heatmap de similaridade, comparação entre disciplinas e dados brutos.</div>
</div>
""", unsafe_allow_html=True)

abrir = st.toggle("Abrir exploração analítica", value=st.session_state.modo_explorar, key="toggle_explorar")
st.session_state.modo_explorar = abrir

if st.session_state.modo_explorar:
    st.markdown("<br>", unsafe_allow_html=True)
    sim_full, coords_full, labels_full, cnames_full = build_analytics(df, 8)
    sim_media = sim_full[np.triu_indices_from(sim_full, k=1)].mean()

    st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-val">{len(df)}</div><div class="kpi-lbl">Disciplinas</div></div>
  <div class="kpi"><div class="kpi-val">{df['School'].nunique()}</div><div class="kpi-lbl">Escolas</div></div>
  <div class="kpi"><div class="kpi-val">{sim_media:.2f}</div><div class="kpi-lbl">Sim. média</div></div>
  <div class="kpi"><div class="kpi-val">8</div><div class="kpi-lbl">Clusters</div></div>
</div>
""", unsafe_allow_html=True)

    tab_mapa, tab_grafo, tab_heat, tab_comp, tab_dados = st.tabs(
        ["🗺 Mapa Temático", "🕸 Grafo", "🔥 Heatmap", "🔍 Comparar disciplinas", "📋 Dados"]
    )

    with tab_mapa:
        n_cl = st.slider("Clusters", 3, 15, 8, key="sl_clusters")
        _, coords_a, labels_a, cnames_a = build_analytics(df, n_cl)
        plot_df = df.copy()
        plot_df["x"] = coords_a[:, 0]
        plot_df["y"] = coords_a[:, 1]
        plot_df["cluster"] = [cnames_a.get(l, str(l)) for l in labels_a]
        plot_df["hover"] = plot_df["Name"] + "<br>" + plot_df["School"] + " | " + plot_df["credits_num"].fillna(0).astype(int).astype(str) + " cr."
        fig = px.scatter(plot_df, x="x", y="y", color="cluster", hover_name="hover",
                         color_discrete_sequence=px.colors.qualitative.Safe, height=500,
                         labels={"x": "Dim 1", "y": "Dim 2", "cluster": "Cluster"})
        fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="white")))
        fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff", font=dict(family="Inter", color="#0f1730"),
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
                          yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            ec = df["School"].value_counts().reset_index()
            ec.columns = ["Escola", "n"]
            fig2 = px.bar(ec, x="n", y="Escola", orientation="h",
                          color="n", color_continuous_scale=["#eef1fb", "#223886"], height=360)
            fig2.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend=False,
                               coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0),
                               font=dict(family="Inter", color="#0f1730"),
                               yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            cd = plot_df["cluster"].value_counts().reset_index()
            cd.columns = ["Cluster", "n"]
            fig3 = px.pie(cd, names="Cluster", values="n",
                          color_discrete_sequence=px.colors.qualitative.Safe, height=360)
            fig3.update_traces(textposition="inside", textinfo="percent+label")
            fig3.update_layout(font=dict(family="Inter", color="#0f1730"),
                               paper_bgcolor="#fff", showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig3, use_container_width=True)

    with tab_grafo:
        grafo = carregar_grafo()
        if grafo is None:
            st.info("Grafo nao encontrado. Execute: `python scripts/build_graph.py`")
        else:
            CORES = px.colors.qualitative.Safe
            nos = grafo["nos"]
            arestas = grafo["arestas"]

            threshold_ui = st.slider("Similaridade minima", 0.25, 0.70, 0.30, 0.05, key="graph_thresh")
            arestas_fil = [a for a in arestas if a["sim"] >= threshold_ui]

            # Usar coordenadas pré-calculadas (spring layout)
            coords = {n["id"]: (n["x"], n["y"]) for n in nos}

            edge_x, edge_y = [], []
            for a in arestas_fil:
                x0, y0 = coords[a["source"]]
                x1, y1 = coords[a["target"]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            cluster_ids = sorted(set(n["cluster"] for n in nos))
            membros_por_cluster = {cid: [n for n in nos if n["cluster"] == cid] for cid in cluster_ids}

            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.6, color="#d0d5e8"),
                hoverinfo="none", showlegend=False,
            ))
            for cid in cluster_ids:
                membros = membros_por_cluster[cid]
                cor = CORES[cid % len(CORES)]
                fig_g.add_trace(go.Scatter(
                    x=[coords[n["id"]][0] for n in membros],
                    y=[coords[n["id"]][1] for n in membros],
                    mode="markers+text",
                    marker=dict(size=[8 + n["grau"] * 0.5 for n in membros], color=cor, opacity=0.85,
                                line=dict(width=0.5, color="#fff")),
                    text=["" if n["grau"] < 10 else n["nome"].split()[0] for n in membros],
                    textposition="top center",
                    textfont=dict(size=9),
                    customdata=[n["nome"] for n in membros],
                    hovertemplate="<b>%{customdata}</b><br>%{meta}<br>Grau: %{marker.size}<extra></extra>",
                    meta=[n["departamento"] for n in membros],
                    name=f"Cluster {cid + 1} ({len(membros)})",
                ))

            fig_g.update_layout(
                height=600,
                paper_bgcolor="#fff", plot_bgcolor="#f8f9fc",
                font=dict(family="Inter", color="#0f1730"),
                showlegend=True,
                legend=dict(orientation="v", x=1.01, y=1, font=dict(size=11)),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"{len(nos)} disciplinas, {len(arestas_fil)} conexoes (sim >= {threshold_ui}), {grafo['n_clusters']} clusters")

    with tab_heat:
        max_d = 40
        if len(df) > max_d:
            st.caption(f"Exibindo os {max_d} mais conectados entre si.")
            top_idx = np.argsort(sim_full.sum(axis=1))[-max_d:]
            sim_d = sim_full[np.ix_(top_idx, top_idx)]
            nomes_d = df.iloc[top_idx]["Name"].str.slice(0, 35).tolist()
        else:
            sim_d, nomes_d = sim_full, df["Name"].str.slice(0, 35).tolist()
        fig_h = go.Figure(go.Heatmap(z=sim_d, x=nomes_d, y=nomes_d,
                                      colorscale=[[0, "#f0f3ff"], [0.5, "#7b93d3"], [1, "#223886"]],
                                      zmin=0, zmax=1, colorbar=dict(title="Sim.", thickness=14)))
        fig_h.update_layout(height=580, plot_bgcolor="#fff", paper_bgcolor="#fff",
                             font=dict(family="Inter", size=9, color="#0f1730"),
                             margin=dict(l=0,r=0,t=0,b=0),
                             xaxis=dict(tickangle=-45), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_h, use_container_width=True)
        pares = [{"Disciplina A": df.iloc[i]["Name"], "Escola A": df.iloc[i]["School"],
                  "Disciplina B": df.iloc[j]["Name"], "Escola B": df.iloc[j]["School"],
                  "Similaridade": round(float(sim_full[i, j]), 3)}
                 for i in range(len(df)) for j in range(i+1, len(df)) if sim_full[i, j] > 0.05]
        if pares:
            pdf = pd.DataFrame(pares).sort_values("Similaridade", ascending=False).head(20)
            st.dataframe(pdf, use_container_width=True,
                         column_config={"Similaridade": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})

    with tab_comp:
        top_n = st.slider("Top similares", 3, 15, 5, key="sl_topn")
        disc_sel = st.selectbox("Selecione uma disciplina", df["Name"].tolist(), key="sel_disc")
        if disc_sel:
            idx_sel = df[df["Name"] == disc_sel].index[0]
            top_idx2 = np.argsort(sim_full[idx_sel])[::-1][1:top_n+1]
            info = df.iloc[idx_sel]
            school = (info.get("School name") or info.get("School") or "").strip()
            st.markdown(f"""
<div class="disc-card" style="margin-bottom:20px;">
  <div class="disc-name">📖 {info['Name']}</div>
  <div class="disc-meta">{school} &nbsp;|&nbsp; {int(info['credits_num'] or 0)} créditos</div>
  <div class="disc-obj">{str(info.get('Objectives',''))[:300]}...</div>
</div>""", unsafe_allow_html=True)
            for rank, i in enumerate(top_idx2, 1):
                sv = sim_full[idx_sel][i]
                row = df.iloc[i]
                pct = int(sv * 100)
                bc = "sim-high" if sv > 0.15 else ("sim-mid" if sv > 0.05 else "sim-low")
                school_r = (row.get("School name") or row.get("School") or "").strip()
                st.markdown(f"""
<div class="disc-card">
  <div class="disc-top">
    <div>
      <div class="disc-name">#{rank} {row['Name']}</div>
      <div class="disc-meta">{school_r} &nbsp;|&nbsp; {int(row['credits_num'] or 0)} cr.</div>
    </div>
    <span class="sim-badge {bc}">{pct}%</span>
  </div>
  <div class="disc-obj">{str(row.get('Objectives',''))[:200]}...</div>
</div>""", unsafe_allow_html=True)

    with tab_dados:
        busca_d = st.text_input("Filtrar", placeholder="nome, escola, tópico...", key="busca_dados")
        df_s = df.copy()
        if busca_d:
            m = (df_s["Name"].str.contains(busca_d, case=False, na=False) |
                 df_s["School"].str.contains(busca_d, case=False, na=False) |
                 df_s["Objectives"].str.contains(busca_d, case=False, na=False))
            df_s = df_s[m]
        cols_show = ["School", "Code", "Name", "credits_num", "Professors", "Start date", "End date"]
        st.dataframe(df_s[cols_show].rename(columns={
            "credits_num": "Créditos", "School": "Escola", "Code": "Código",
            "Name": "Nome", "Professors": "Professores", "Start date": "Início", "End date": "Fim"
        }), use_container_width=True, height=420)
        st.caption(f"{len(df_s)} disciplinas")
