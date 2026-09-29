# -*- coding: utf-8 -*-
"""
Graficos_Tabelas_TCC_13.py

Gera os principais gráficos e tabelas da avaliação pareada do TCC,
comparando os cenários ORIGINAL e INDEXADO.

Arquivos esperados na mesma pasta:
    avaliacao_pareada_original_por_consulta.csv
    avaliacao_pareada_original_detalhado.csv
    avaliacao_pareada_indexado_por_consulta.csv
    avaliacao_pareada_indexado_detalhado.csv

Compatibilidade:
Se os arquivos do cenário indexado ainda estiverem com os nomes "(1)"
gerados anteriormente, o script consegue usá-los automaticamente:
    avaliacao_pareada_original_por_consulta(1).csv
    avaliacao_pareada_original_detalhado(1).csv

Nesse caso, o arquivo sem "(1)" é tratado como ORIGINAL e o arquivo
com "(1)" como INDEXADO, independentemente do conteúdo da coluna cenario.

Dependências:
    pip install pandas numpy matplotlib

Execução:
    python Graficos_Tabelas_TCC_13.py

Opcionalmente, é possível informar os arquivos manualmente:
    python Graficos_Tabelas_TCC_13.py ^
      --original-por-consulta avaliacao_pareada_original_por_consulta.csv ^
      --indexado-por-consulta avaliacao_pareada_indexado_por_consulta.csv ^
      --original-detalhado avaliacao_pareada_original_detalhado.csv ^
      --indexado-detalhado avaliacao_pareada_indexado_detalhado.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METODOS = ["DDQN", "DQN", "Random", "Greedy", "Qwen"]
CENARIOS = ["Original", "Indexado"]
TOL = 1e-9


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera gráficos e tabelas LaTeX da avaliação pareada."
    )
    parser.add_argument("--original-por-consulta", default=None)
    parser.add_argument("--indexado-por-consulta", default=None)
    parser.add_argument("--original-detalhado", default=None)
    parser.add_argument("--indexado-detalhado", default=None)
    parser.add_argument("--saida", default="saida_tcc_pareada")
    return parser.parse_args()


def localizar_arquivos(base: Path, args):
    """
    Prioridade:
    1) caminhos informados por linha de comando;
    2) nomes canônicos original/indexado;
    3) fallback histórico: arquivos '(1)' tratados como indexados.
    """
    original_pc = Path(args.original_por_consulta) if args.original_por_consulta else None
    indexado_pc = Path(args.indexado_por_consulta) if args.indexado_por_consulta else None
    original_det = Path(args.original_detalhado) if args.original_detalhado else None
    indexado_det = Path(args.indexado_detalhado) if args.indexado_detalhado else None

    if original_pc is None:
        original_pc = base / "avaliacao_pareada_original_por_consulta.csv"
    if original_det is None:
        original_det = base / "avaliacao_pareada_original_detalhado.csv"

    if indexado_pc is None:
        candidato = base / "avaliacao_pareada_indexado_por_consulta.csv"
        if candidato.exists():
            indexado_pc = candidato
        else:
            indexado_pc = base / "avaliacao_pareada_original_por_consulta(1).csv"

    if indexado_det is None:
        candidato = base / "avaliacao_pareada_indexado_detalhado.csv"
        if candidato.exists():
            indexado_det = candidato
        else:
            indexado_det = base / "avaliacao_pareada_original_detalhado(1).csv"

    arquivos = {
        "original_por_consulta": original_pc,
        "indexado_por_consulta": indexado_pc,
        "original_detalhado": original_det,
        "indexado_detalhado": indexado_det,
    }

    faltantes = [f"{k}: {v}" for k, v in arquivos.items() if not v.exists()]
    if faltantes:
        raise FileNotFoundError(
            "Não encontrei todos os quatro CSVs necessários:\n  - "
            + "\n  - ".join(faltantes)
            + "\n\nRenomeie os arquivos ou informe os caminhos pela linha de comando."
        )

    return arquivos


def carregar_dados(arquivos):
    orig_pc = pd.read_csv(arquivos["original_por_consulta"])
    idx_pc = pd.read_csv(arquivos["indexado_por_consulta"])
    orig_det = pd.read_csv(arquivos["original_detalhado"])
    idx_det = pd.read_csv(arquivos["indexado_detalhado"])

    # O rótulo do cenário é imposto pelo arquivo usado, porque uma execução
    # anterior do banco indexado pode ter sido salva ainda com cenario="original".
    orig_pc["cenario"] = "Original"
    idx_pc["cenario"] = "Indexado"
    orig_det["cenario"] = "Original"
    idx_det["cenario"] = "Indexado"

    validar_colunas(orig_pc, "por_consulta")
    validar_colunas(idx_pc, "por_consulta")
    validar_colunas(orig_det, "detalhado")
    validar_colunas(idx_det, "detalhado")

    return orig_pc, idx_pc, orig_det, idx_det


def validar_colunas(df, tipo):
    if tipo == "por_consulta":
        obrigatorias = {
            "cenario", "metodo", "consulta_nome", "consulta_hash",
            "qtd_tabelas", "custo_original", "custo_metodo",
            "melhoria_percentual", "usou_fallback", "qtd_fallback"
        }
    else:
        obrigatorias = {
            "cenario", "metodo", "consulta_nome", "consulta_hash",
            "qtd_tabelas", "custo_original", "custo_final"
        }

    faltantes = obrigatorias - set(df.columns)
    if faltantes:
        raise ValueError(
            f"CSV {tipo} sem as colunas obrigatórias: {sorted(faltantes)}"
        )


def ordenar_metodos(df):
    df = df.copy()
    df["metodo"] = pd.Categorical(
        df["metodo"], categories=METODOS, ordered=True
    )
    return df.sort_values("metodo").reset_index(drop=True)


def resumo_metodos(df_pc: pd.DataFrame, somente_3mais=False):
    base = df_pc.copy()
    if somente_3mais:
        base = base[base["qtd_tabelas"] >= 3].copy()

    linhas = []
    for metodo in METODOS:
        d = base[base["metodo"] == metodo].copy()
        if d.empty:
            continue

        custo_original_total = d["custo_original"].sum()
        custo_metodo_total = d["custo_metodo"].sum()

        melhoria_global = (
            (custo_original_total - custo_metodo_total)
            / custo_original_total * 100.0
            if custo_original_total != 0 else np.nan
        )

        imp = d["melhoria_percentual"].astype(float)
        vitorias = int((imp > TOL).sum())
        empates = int((imp.abs() <= TOL).sum())
        derrotas = int((imp < -TOL).sum())

        fallback = d["usou_fallback"].fillna(False).astype(bool)
        qtd_fallback = pd.to_numeric(d["qtd_fallback"], errors="coerce").fillna(0)

        linhas.append({
            "cenario": d["cenario"].iloc[0],
            "metodo": metodo,
            "n_consultas": d["consulta_hash"].nunique(),
            "custo_medio": d["custo_metodo"].mean(),
            "custo_total": custo_metodo_total,
            "custo_original_total": custo_original_total,
            "melhoria_global_pct": melhoria_global,
            "melhoria_mediana_pct": imp.median(),
            "melhoria_media_simples_pct": imp.mean(),
            "vitorias": vitorias,
            "empates": empates,
            "derrotas": derrotas,
            "fallback_consultas": int(fallback.sum()) if metodo == "Qwen" else 0,
            "fallback_pct": float(fallback.mean() * 100.0) if metodo == "Qwen" else 0.0,
            "fallback_acoes": int(qtd_fallback.sum()) if metodo == "Qwen" else 0,
        })

    return ordenar_metodos(pd.DataFrame(linhas))


def format_num_br(x, casas=2):
    if pd.isna(x):
        return "--"
    s = f"{float(x):,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct_br(x, casas=2):
    if pd.isna(x):
        return "--"
    return format_num_br(x, casas) + r"\%"


def latex_escape(text):
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def salvar_tabela_latex(
    df,
    caminho,
    colunas,
    cabecalhos,
    alinhamento,
    caption,
    label,
    formatadores=None,
    footnote=None,
):
    formatadores = formatadores or {}

    linhas = []
    linhas.append(r"\begin{table}[htbp]")
    linhas.append(r"\centering")
    linhas.append(r"\caption{" + caption + "}")
    linhas.append(r"\label{" + label + "}")
    linhas.append(r"\small")
    linhas.append(r"\begin{tabular}{" + alinhamento + "}")
    linhas.append(r"\toprule")
    linhas.append(" & ".join(cabecalhos) + r" \\")
    linhas.append(r"\midrule")

    for _, row in df.iterrows():
        vals = []
        for col in colunas:
            val = row[col]
            if col in formatadores:
                vals.append(formatadores[col](val))
            else:
                vals.append(latex_escape(val))
        linhas.append(" & ".join(vals) + r" \\")

    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}")
    if footnote:
        linhas.append(r"\vspace{2mm}")
        linhas.append(r"\parbox{0.96\linewidth}{\footnotesize " + footnote + "}")
    linhas.append(r"\end{table}")
    linhas.append("")

    caminho.write_text("\n".join(linhas), encoding="utf-8")


def salvar_figura(fig, pasta, nome):
    png = pasta / f"{nome}.png"
    pdf = pasta / f"{nome}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------

def gerar_tabelas(orig_pc, idx_pc, pasta_tabelas):
    todos = pd.concat([orig_pc, idx_pc], ignore_index=True)

    resumo_orig = resumo_metodos(orig_pc)
    resumo_idx = resumo_metodos(idx_pc)
    resumo_3_orig = resumo_metodos(orig_pc, somente_3mais=True)
    resumo_3_idx = resumo_metodos(idx_pc, somente_3mais=True)

    resumo_total = pd.concat([resumo_orig, resumo_idx], ignore_index=True)
    resumo_total.to_csv(
        pasta_tabelas / "tabela_01_resumo_completo.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Tabela principal: comparação dos dois cenários por método
    comp = resumo_orig.merge(
        resumo_idx,
        on="metodo",
        suffixes=("_original", "_indexado")
    )

    comp["delta_melhoria_pp"] = (
        comp["melhoria_global_pct_indexado"]
        - comp["melhoria_global_pct_original"]
    )
    comp["variacao_custo_metodo_pct"] = (
        (comp["custo_medio_original"] - comp["custo_medio_indexado"])
        / comp["custo_medio_original"] * 100.0
    )

    comp.to_csv(
        pasta_tabelas / "tabela_02_comparacao_original_indexado.csv",
        index=False,
        encoding="utf-8-sig"
    )

    salvar_tabela_latex(
        comp,
        pasta_tabelas / "tabela_02_comparacao_original_indexado.tex",
        colunas=[
            "metodo",
            "custo_medio_original",
            "melhoria_global_pct_original",
            "custo_medio_indexado",
            "melhoria_global_pct_indexado",
            "delta_melhoria_pp",
        ],
        cabecalhos=[
            "Método",
            r"Custo médio\\Original",
            r"Melhoria\\Original",
            r"Custo médio\\Indexado",
            r"Melhoria\\Indexado",
            r"$\Delta$ (p.p.)",
        ],
        alinhamento="lrrrrr",
        caption=(
            "Comparação do custo médio e da melhoria global das estratégias "
            "nos cenários original e indexado."
        ),
        label="tab:comparacao_original_indexado",
        formatadores={
            "custo_medio_original": lambda x: format_num_br(x, 0),
            "melhoria_global_pct_original": lambda x: format_pct_br(x, 2),
            "custo_medio_indexado": lambda x: format_num_br(x, 0),
            "melhoria_global_pct_indexado": lambda x: format_pct_br(x, 2),
            "delta_melhoria_pp": lambda x: format_num_br(x, 2),
        },
        footnote=(
            "A melhoria global é calculada pela razão entre a soma dos custos "
            "originais e a soma dos custos produzidos pelo método. Valores de "
            "$\\Delta$ negativos indicam redução da vantagem relativa do método "
            "após a indexação."
        )
    )

    # Tabelas de consistência por cenário
    for nome, resumo in [("original", resumo_orig), ("indexado", resumo_idx)]:
        resumo.to_csv(
            pasta_tabelas / f"tabela_03_resumo_{nome}.csv",
            index=False,
            encoding="utf-8-sig"
        )
        salvar_tabela_latex(
            resumo,
            pasta_tabelas / f"tabela_03_resumo_{nome}.tex",
            colunas=[
                "metodo", "melhoria_global_pct", "melhoria_mediana_pct",
                "vitorias", "empates", "derrotas"
            ],
            cabecalhos=[
                "Método", "Melhoria global", "Mediana",
                "Vitórias", "Empates", "Derrotas"
            ],
            alinhamento="lrrrrr",
            caption=(
                f"Desempenho das estratégias no cenário {nome}, "
                "considerando as consultas da avaliação pareada."
            ),
            label=f"tab:resumo_{nome}",
            formatadores={
                "melhoria_global_pct": lambda x: format_pct_br(x, 2),
                "melhoria_mediana_pct": lambda x: format_pct_br(x, 2),
                "vitorias": lambda x: str(int(x)),
                "empates": lambda x: str(int(x)),
                "derrotas": lambda x: str(int(x)),
            },
        )

    # Resumo apenas para consultas com >= 3 tabelas
    resumo_3 = pd.concat([resumo_3_orig, resumo_3_idx], ignore_index=True)
    resumo_3.to_csv(
        pasta_tabelas / "tabela_04_resumo_consultas_3_ou_mais_tabelas.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Impacto da indexação na SQL de referência por consulta
    orig_ref = (
        orig_pc[["consulta_hash", "consulta_nome", "qtd_tabelas", "custo_original"]]
        .drop_duplicates("consulta_hash")
        .rename(columns={"custo_original": "custo_original_base_original"})
    )
    idx_ref = (
        idx_pc[["consulta_hash", "consulta_nome", "qtd_tabelas", "custo_original"]]
        .drop_duplicates("consulta_hash")
        .rename(columns={"custo_original": "custo_original_base_indexada"})
    )

    impacto = orig_ref.merge(
        idx_ref,
        on="consulta_hash",
        suffixes=("_orig", "_idx")
    )
    impacto["consulta_nome"] = impacto["consulta_nome_orig"]
    impacto["qtd_tabelas"] = impacto["qtd_tabelas_orig"]
    impacto["impacto_indexacao_pct"] = (
        (impacto["custo_original_base_original"]
         - impacto["custo_original_base_indexada"])
        / impacto["custo_original_base_original"] * 100.0
    )
    impacto = impacto.sort_values("consulta_nome").reset_index(drop=True)

    impacto[
        [
            "consulta_nome",
            "qtd_tabelas",
            "custo_original_base_original",
            "custo_original_base_indexada",
            "impacto_indexacao_pct",
        ]
    ].to_csv(
        pasta_tabelas / "tabela_05_impacto_indexacao_por_consulta.csv",
        index=False,
        encoding="utf-8-sig"
    )

    salvar_tabela_latex(
        impacto,
        pasta_tabelas / "tabela_05_impacto_indexacao_por_consulta.tex",
        colunas=[
            "consulta_nome", "qtd_tabelas",
            "custo_original_base_original",
            "custo_original_base_indexada",
            "impacto_indexacao_pct"
        ],
        cabecalhos=[
            "Consulta", "Tabelas", "Custo original",
            "Custo indexado", "Efeito da indexação"
        ],
        alinhamento="lrrrr",
        caption=(
            "Impacto da indexação sobre o custo estimado das consultas "
            "originais do benchmark."
        ),
        label="tab:impacto_indexacao_consulta",
        formatadores={
            "qtd_tabelas": lambda x: str(int(x)),
            "custo_original_base_original": lambda x: format_num_br(x, 0),
            "custo_original_base_indexada": lambda x: format_num_br(x, 0),
            "impacto_indexacao_pct": lambda x: format_pct_br(x, 2),
        },
        footnote=(
            "Valores positivos indicam redução do custo estimado após a "
            "indexação; valores negativos indicam aumento do custo estimado."
        )
    )

    # Tabela Qwen fallback
    qwen = resumo_total[resumo_total["metodo"] == "Qwen"].copy()
    qwen.to_csv(
        pasta_tabelas / "tabela_06_qwen_fallback.csv",
        index=False,
        encoding="utf-8-sig"
    )
    salvar_tabela_latex(
        qwen,
        pasta_tabelas / "tabela_06_qwen_fallback.tex",
        colunas=[
            "cenario", "n_consultas", "fallback_consultas",
            "fallback_pct", "fallback_acoes"
        ],
        cabecalhos=[
            "Cenário", "Consultas", "Com fallback",
            "Fallback", "Ações fallback"
        ],
        alinhamento="lrrrr",
        caption="Utilização do mecanismo de fallback pelo baseline Qwen.",
        label="tab:qwen_fallback",
        formatadores={
            "n_consultas": lambda x: str(int(x)),
            "fallback_consultas": lambda x: str(int(x)),
            "fallback_pct": lambda x: format_pct_br(x, 2),
            "fallback_acoes": lambda x: str(int(x)),
        },
    )

    # Tabela detalhada por consulta e método, útil para apêndice ou conferência
    detalhe_consulta = todos[
        [
            "cenario", "metodo", "consulta_nome", "qtd_tabelas",
            "custo_original", "custo_metodo", "melhoria_percentual"
        ]
    ].copy()

    detalhe_consulta.to_csv(
        pasta_tabelas / "tabela_07_resultado_por_consulta_metodo.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "resumo_original": resumo_orig,
        "resumo_indexado": resumo_idx,
        "comparacao": comp,
        "impacto": impacto,
    }


# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------

def gerar_graficos(orig_pc, idx_pc, tabelas, pasta_graficos):
    resumo_orig = tabelas["resumo_original"]
    resumo_idx = tabelas["resumo_indexado"]
    impacto = tabelas["impacto"]

    # 1) Custo médio por método - Original x Indexado
    x = np.arange(len(METODOS))
    largura = 0.36

    y_orig = [
        float(resumo_orig.loc[resumo_orig["metodo"] == m, "custo_medio"].iloc[0])
        for m in METODOS
    ]
    y_idx = [
        float(resumo_idx.loc[resumo_idx["metodo"] == m, "custo_medio"].iloc[0])
        for m in METODOS
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - largura/2, y_orig, largura, label="Original")
    bars2 = ax.bar(x + largura/2, y_idx, largura, label="Indexado")
    ax.set_ylabel("Custo médio estimado")
    ax.set_xlabel("Método")
    ax.set_title("Custo médio estimado por método e cenário")
    ax.set_xticks(x)
    ax.set_xticklabels(METODOS)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars1, bars2):
        for b in bars:
            ax.annotate(
                f"{b.get_height()/1_000_000:.2f}M",
                (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=8,
                xytext=(0, 3), textcoords="offset points"
            )

    salvar_figura(fig, pasta_graficos, "figura_01_custo_medio_original_vs_indexado")

    # 2) Melhoria global por método - Original x Indexado
    imp_orig = [
        float(resumo_orig.loc[
            resumo_orig["metodo"] == m, "melhoria_global_pct"
        ].iloc[0])
        for m in METODOS
    ]
    imp_idx = [
        float(resumo_idx.loc[
            resumo_idx["metodo"] == m, "melhoria_global_pct"
        ].iloc[0])
        for m in METODOS
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - largura/2, imp_orig, largura, label="Original")
    bars2 = ax.bar(x + largura/2, imp_idx, largura, label="Indexado")
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Melhoria global (%)")
    ax.set_xlabel("Método")
    ax.set_title("Melhoria global em relação à SQL original")
    ax.set_xticks(x)
    ax.set_xticklabels(METODOS)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars1, bars2):
        for b in bars:
            ax.annotate(
                f"{b.get_height():.1f}%",
                (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center",
                va="bottom" if b.get_height() >= 0 else "top",
                fontsize=8,
                xytext=(0, 3 if b.get_height() >= 0 else -3),
                textcoords="offset points"
            )

    salvar_figura(fig, pasta_graficos, "figura_02_melhoria_global_original_vs_indexado")

    # 3) Vitórias / Empates / Derrotas
    resumo_ved = pd.concat([resumo_orig, resumo_idx], ignore_index=True)
    labels = [
        f"{row.metodo}\n{row.cenario}"
        for row in resumo_ved.itertuples(index=False)
    ]
    vit = resumo_ved["vitorias"].to_numpy()
    emp = resumo_ved["empates"].to_numpy()
    der = resumo_ved["derrotas"].to_numpy()

    xx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(xx, vit, label="Vitórias")
    ax.bar(xx, emp, bottom=vit, label="Empates")
    ax.bar(xx, der, bottom=vit + emp, label="Derrotas")
    ax.set_ylabel("Número de consultas")
    ax.set_xlabel("Método e cenário")
    ax.set_title("Consistência das estratégias: vitórias, empates e derrotas")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, max(resumo_ved["n_consultas"]) + 2)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    salvar_figura(fig, pasta_graficos, "figura_03_vitorias_empates_derrotas")

    # 4) Efeito da indexação na SQL original por consulta
    imp = impacto.copy()
    imp = imp.sort_values("impacto_indexacao_pct", ascending=False)
    xx = np.arange(len(imp))

    fig, ax = plt.subplots(figsize=(12, 5.8))
    bars = ax.bar(xx, imp["impacto_indexacao_pct"])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Redução do custo após indexação (%)")
    ax.set_xlabel("Consulta")
    ax.set_title("Impacto da indexação sobre a SQL original por consulta")
    ax.set_xticks(xx)
    ax.set_xticklabels(imp["consulta_nome"], rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.25)

    # Anota somente valores muito expressivos para não poluir o gráfico
    for b, valor in zip(bars, imp["impacto_indexacao_pct"]):
        if abs(valor) >= 10:
            ax.annotate(
                f"{valor:.1f}%",
                (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center",
                va="bottom" if valor >= 0 else "top",
                fontsize=7,
                xytext=(0, 3 if valor >= 0 else -3),
                textcoords="offset points"
            )

    salvar_figura(fig, pasta_graficos, "figura_04_impacto_indexacao_por_consulta")

    # 5) Melhoria por consulta para DQN e DDQN - cenário original
    #     Útil para mostrar onde os dois métodos aprendidos divergem.
    for cenario_nome, df in [("Original", orig_pc), ("Indexado", idx_pc)]:
        d = df[df["metodo"].isin(["DQN", "DDQN"])].copy()
        pivot = d.pivot(
            index="consulta_nome",
            columns="metodo",
            values="melhoria_percentual"
        )
        pivot = pivot.reindex(sorted(pivot.index))

        xx = np.arange(len(pivot))
        fig, ax = plt.subplots(figsize=(12, 5.8))
        ax.plot(xx, pivot["DQN"], marker="o", label="DQN")
        ax.plot(xx, pivot["DDQN"], marker="o", label="DDQN")
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Melhoria por consulta (%)")
        ax.set_xlabel("Consulta")
        ax.set_title(f"DQN × DDQN — melhoria por consulta ({cenario_nome})")
        ax.set_xticks(xx)
        ax.set_xticklabels(pivot.index, rotation=60, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        salvar_figura(
            fig,
            pasta_graficos,
            f"figura_05_dqn_ddqn_por_consulta_{cenario_nome.lower()}"
        )


# ---------------------------------------------------------------------
# Relatório textual auxiliar
# ---------------------------------------------------------------------

def gerar_resumo_textual(tabelas, pasta_saida):
    comp = tabelas["comparacao"].copy()
    linhas = []
    linhas.append("RESUMO AUTOMÁTICO - AVALIAÇÃO PAREADA")
    linhas.append("=" * 70)
    linhas.append("")
    linhas.append("Melhoria global por método:")
    for _, r in comp.iterrows():
        linhas.append(
            f"- {r['metodo']}: "
            f"Original={r['melhoria_global_pct_original']:.4f}% | "
            f"Indexado={r['melhoria_global_pct_indexado']:.4f}% | "
            f"Delta={r['delta_melhoria_pp']:.4f} p.p."
        )

    linhas.append("")
    linhas.append(
        "Observação: a melhoria global usa a soma dos custos, evitando que "
        "consultas de custo original muito baixo dominem a análise por meio "
        "de percentuais individuais extremos."
    )

    (pasta_saida / "resumo_automatico.txt").write_text(
        "\n".join(linhas), encoding="utf-8"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    base = Path.cwd()
    saida = base / args.saida
    pasta_graficos = saida / "graficos"
    pasta_tabelas = saida / "tabelas"

    pasta_graficos.mkdir(parents=True, exist_ok=True)
    pasta_tabelas.mkdir(parents=True, exist_ok=True)

    arquivos = localizar_arquivos(base, args)

    print("=" * 72)
    print("GERAÇÃO DE GRÁFICOS E TABELAS - AVALIAÇÃO PAREADA")
    print("=" * 72)
    for nome, caminho in arquivos.items():
        print(f"{nome:28s}: {caminho.name}")

    if "(1)" in arquivos["indexado_por_consulta"].name:
        print(
            "\nATENÇÃO: o arquivo '(1)' foi interpretado como cenário INDEXADO. "
            "O conteúdo da coluna 'cenario' será sobrescrito em memória."
        )

    orig_pc, idx_pc, orig_det, idx_det = carregar_dados(arquivos)

    tabelas = gerar_tabelas(
        orig_pc=orig_pc,
        idx_pc=idx_pc,
        pasta_tabelas=pasta_tabelas
    )
    gerar_graficos(
        orig_pc=orig_pc,
        idx_pc=idx_pc,
        tabelas=tabelas,
        pasta_graficos=pasta_graficos
    )
    gerar_resumo_textual(tabelas, saida)

    print("\nArquivos gerados com sucesso.")
    print(f"Pasta principal: {saida.resolve()}")
    print(f"Gráficos:        {pasta_graficos.resolve()}")
    print(f"Tabelas:         {pasta_tabelas.resolve()}")
    print("\nPrincipais arquivos para o TCC:")
    print("  - figura_01_custo_medio_original_vs_indexado.(png/pdf)")
    print("  - figura_02_melhoria_global_original_vs_indexado.(png/pdf)")
    print("  - figura_03_vitorias_empates_derrotas.(png/pdf)")
    print("  - figura_04_impacto_indexacao_por_consulta.(png/pdf)")
    print("  - figura_05_dqn_ddqn_por_consulta_original.(png/pdf)")
    print("  - figura_05_dqn_ddqn_por_consulta_indexado.(png/pdf)")
    print("  - tabela_02_comparacao_original_indexado.tex")
    print("  - tabela_03_resumo_original.tex")
    print("  - tabela_03_resumo_indexado.tex")
    print("  - tabela_05_impacto_indexacao_por_consulta.tex")
    print("  - tabela_06_qwen_fallback.tex")


if __name__ == "__main__":
    main()
