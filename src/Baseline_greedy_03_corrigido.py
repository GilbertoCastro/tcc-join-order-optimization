import csv
import random
from datetime import datetime

import numpy as np

from LearningJob_v4_01_corrigido import LearningJob


# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
EPISODIOS = 500
SEED = 42

ARQUIVO_SAIDA = "resultados_greedy_baseline.csv"


# =====================================================================
# REPRODUTIBILIDADE
# =====================================================================
def configurar_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


# =====================================================================
# VALIDAÇÃO DE CONTINUIDADE DO GRAFO
# =====================================================================
def ainda_tem_caminho_para_finalizar(env: LearningJob, action_idx: int) -> bool:
    """
    Simula a união produzida pela ação candidata e verifica se os componentes
    restantes continuam formando um grafo conectado.

    Essa verificação preserva a ideia do baseline Greedy original:
    evitar uma escolha local que inviabilize a construção completa da árvore
    de JOIN.
    """
    componentes = [set(c) for c in env.components]

    left_alias, right_alias = env.action_pairs[action_idx]

    try:
        comp_left_idx = env._find_component_index(left_alias)
        comp_right_idx = env._find_component_index(right_alias)
    except ValueError:
        return False

    if comp_left_idx == comp_right_idx:
        return False

    # Une os dois componentes na cópia.
    novo_componente = (
        componentes[comp_left_idx]
        | componentes[comp_right_idx]
    )

    for idx in sorted(
        [comp_left_idx, comp_right_idx],
        reverse=True,
    ):
        del componentes[idx]

    componentes.append(novo_componente)

    if len(componentes) == 1:
        return True

    # Cria um grafo entre COMPONENTES, usando apenas JOINs relacionais reais.
    grafo_componentes = {
        i: set()
        for i in range(len(componentes))
    }

    for a, b in env.join_edges:
        comp_a = None
        comp_b = None

        for idx, comp in enumerate(componentes):
            if a in comp:
                comp_a = idx
            if b in comp:
                comp_b = idx

        if (
            comp_a is not None
            and comp_b is not None
            and comp_a != comp_b
        ):
            grafo_componentes[comp_a].add(comp_b)
            grafo_componentes[comp_b].add(comp_a)

    # Busca em profundidade para verificar conectividade.
    visitados = set()
    pilha = [0]

    while pilha:
        atual = pilha.pop()

        if atual in visitados:
            continue

        visitados.add(atual)
        pilha.extend(
            grafo_componentes[atual] - visitados
        )

    return len(visitados) == len(componentes)


# =====================================================================
# POLÍTICA GREEDY
# =====================================================================
def escolher_acao_greedy(env: LearningJob):
    """
    Escolhe uma ação Greedy entre as ações válidas.

    Heurística preservada do baseline original:
        score = |componente_esquerdo| + |componente_direito|

    A ação com MAIOR score é selecionada, isto é, a estratégia prioriza
    a união de componentes maiores.

    IMPORTANTE:
    Este Greedy NÃO consulta o custo do PostgreSQL para escolher a próxima
    ação. Portanto, "Greedy" aqui significa ótimo local segundo essa
    heurística estrutural, e não segundo o Total Cost do PostgreSQL.
    """
    candidatas = []

    # Ações estruturalmente válidas segundo o ambiente corrigido.
    for idx in env.valid_action_indices():
        left_alias, right_alias = env.action_pairs[idx]

        try:
            comp_left_idx = env._find_component_index(left_alias)
            comp_right_idx = env._find_component_index(right_alias)
        except ValueError:
            continue

        if comp_left_idx == comp_right_idx:
            continue

        if not ainda_tem_caminho_para_finalizar(env, idx):
            continue

        score = (
            len(env.components[comp_left_idx])
            + len(env.components[comp_right_idx])
        )

        # O segundo elemento negativo faz o desempate pelo menor índice
        # de ação, tornando a política determinística para uma mesma consulta.
        candidatas.append((score, -idx, idx))

    if not candidatas:
        return None

    candidatas.sort(reverse=True)
    return candidatas[0][2]


# =====================================================================
# EXECUÇÃO DO BASELINE
# =====================================================================
def executar_baseline():
    configurar_seed(SEED)

    print("=" * 70)
    print("BASELINE GREEDY - ORDENAÇÃO DE JOINS")
    print("=" * 70)
    print(f"Episódios: {EPISODIOS}")
    print(f"Seed: {SEED}")
    print(
        "Heurística: maximizar "
        "|componente_esquerdo| + |componente_direito|"
    )

    env = LearningJob()
    resultados = []

    try:
        for episodio in range(1, EPISODIOS + 1):

            # O primeiro reset inicializa explicitamente o gerador do Gymnasium.
            # Nos episódios seguintes, o mesmo gerador continua sua sequência,
            # garantindo amostragem reprodutível sem repetir sempre a mesma consulta.
            if episodio == 1:
                state, info_reset = env.reset(seed=SEED)
            else:
                state, info_reset = env.reset()

            terminated = False
            truncated = False

            recompensa_total = 0.0
            passos = 0

            info_final = {}

            while not terminated and not truncated:
                action = escolher_acao_greedy(env)

                if action is None:
                    recompensa_total -= 1000.0
                    truncated = True

                    info_final = {
                        "erro": "Nenhuma ação Greedy válida restante",
                        "custo_estimado_final": float("inf"),
                        "cardinalidade_estimada_final": float("inf"),
                        "consulta_sql_reordenada": "",
                        "ordem_logica_joins": [
                            env.action_pairs[i]
                            for i in env.action_history
                        ],
                    }
                    break

                state, reward, terminated, truncated, step_info = env.step(action)

                recompensa_total += float(reward)
                passos += 1

                if step_info:
                    info_final.update(step_info)

            custo_original = info_reset.get("custo_postgresql")
            custo_final = info_final.get("custo_estimado_final")

            melhoria_percentual = None

            if (
                custo_original is not None
                and custo_final is not None
                and np.isfinite(custo_original)
                and np.isfinite(custo_final)
                and custo_original != 0
            ):
                melhoria_percentual = (
                    (custo_original - custo_final)
                    / custo_original
                    * 100.0
                )

            resultados.append({
                "episodio": episodio,
                "data_execucao": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "consulta": info_reset.get(
                    "consulta_selecionada",
                    "",
                ),

                "aliases": str(
                    info_reset.get("aliases", {})
                ),

                "joins_validos": str(
                    info_reset.get("joins_validos", [])
                ),

                "condicoes_join": str(
                    info_reset.get("condicoes_join", {})
                ),

                "pares_acao": str(
                    info_reset.get("action_pairs", [])
                ),

                # Ordem REAL de execução das ações.
                "ordem_logica_joins": str(
                    info_final.get(
                        "ordem_logica_joins",
                        [
                            env.action_pairs[i]
                            for i in env.action_history
                        ],
                    )
                ),

                # SQL realmente submetida ao EXPLAIN.
                "consulta_sql_reordenada": info_final.get(
                    "consulta_sql_reordenada",
                    "",
                ),

                "custo_postgresql_original": custo_original,

                "cardinalidade_postgresql_original": info_reset.get(
                    "cardinalidade_postgresql"
                ),

                "custo_estimado_final": custo_final,

                "cardinalidade_estimada_final": info_final.get(
                    "cardinalidade_estimada_final"
                ),

                "melhoria_percentual_vs_original": melhoria_percentual,

                "recompensa_total": recompensa_total,
                "passos": passos,

                "finalizado": terminated,
                "truncado": truncated,

                "erro": info_final.get("erro", ""),
            })

            custo_texto = (
                f"{custo_final:.2f}"
                if custo_final is not None
                and np.isfinite(custo_final)
                else "inf/None"
            )

            melhoria_texto = (
                f"{melhoria_percentual:.4f}%"
                if melhoria_percentual is not None
                else "N/A"
            )

            print(
                f"Episódio {episodio:03d} | "
                f"passos={passos} | "
                f"finalizado={terminated} | "
                f"truncado={truncated} | "
                f"reward={recompensa_total:.6f} | "
                f"custo={custo_texto} | "
                f"melhoria={melhoria_texto}"
            )

    finally:
        env.close()

    if not resultados:
        print("Nenhum resultado foi gerado.")
        return

    # =================================================================
    # SALVA CSV
    # =================================================================
    campos = list(resultados[0].keys())

    with open(
        ARQUIVO_SAIDA,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(resultados)

    print("\n" + "=" * 70)
    print("BASELINE GREEDY FINALIZADO")
    print("=" * 70)

    print(f"CSV salvo: {ARQUIVO_SAIDA}")

    finalizados = sum(
        bool(r["finalizado"])
        for r in resultados
    )

    custos_validos = [
        r["custo_estimado_final"]
        for r in resultados
        if r["custo_estimado_final"] is not None
        and np.isfinite(r["custo_estimado_final"])
    ]

    melhorias_validas = [
        r["melhoria_percentual_vs_original"]
        for r in resultados
        if r["melhoria_percentual_vs_original"] is not None
    ]

    print(
        f"Episódios finalizados: "
        f"{finalizados}/{len(resultados)}"
    )

    if custos_validos:
        print(
            f"Custo médio final: "
            f"{np.mean(custos_validos):.6f}"
        )

    if melhorias_validas:
        print(
            f"Melhoria média vs SQL original: "
            f"{np.mean(melhorias_validas):.6f}%"
        )

        print(
            f"Mediana da melhoria vs SQL original: "
            f"{np.median(melhorias_validas):.6f}%"
        )


if __name__ == "__main__":
    executar_baseline()
