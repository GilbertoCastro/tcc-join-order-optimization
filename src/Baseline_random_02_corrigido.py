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

ARQUIVO_SAIDA = "resultados_random_baseline.csv"


# =====================================================================
# REPRODUTIBILIDADE
# =====================================================================
def configurar_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


# =====================================================================
# ESCOLHA ALEATÓRIA ENTRE AÇÕES VÁLIDAS
# =====================================================================
def escolher_acao_valida(env: LearningJob):
    """
    Escolhe uniformemente uma ação dentre as ações atualmente válidas.

    A validação estrutural fica centralizada no próprio ambiente:
    - não permite ciclo;
    - não permite CROSS JOIN;
    - só permite arestas com condição relacional;
    - só conecta componentes distintos.
    """
    acoes_validas = env.valid_action_indices()

    if not acoes_validas:
        return None

    return random.choice(acoes_validas)


# =====================================================================
# EXECUÇÃO DO BASELINE RANDOM
# =====================================================================
def executar_baseline():
    configurar_seed(SEED)

    print("=" * 70)
    print("BASELINE RANDOM - ORDENAÇÃO DE JOINS")
    print("=" * 70)
    print(f"Episódios: {EPISODIOS}")
    print(f"Seed: {SEED}")

    env = LearningJob()
    resultados = []

    try:
        for episodio in range(1, EPISODIOS + 1):
            # Para este baseline de desenvolvimento, a consulta é sorteada
            # pelo próprio ambiente. A avaliação FINAL do TCC será pareada
            # e usará query_index fixo por consulta para todos os métodos.
            state, info_reset = env.reset()

            terminated = False
            truncated = False

            recompensa_total = 0.0
            passos = 0

            info_final = {}

            while not terminated and not truncated:
                action = escolher_acao_valida(env)

                if action is None:
                    recompensa_total -= 1000.0
                    truncated = True

                    info_final = {
                        "erro": "Nenhuma ação válida restante",
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
                "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

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

                # Ordem REAL em que as ações foram executadas.
                "ordem_logica_joins": str(
                    info_final.get(
                        "ordem_logica_joins",
                        [
                            env.action_pairs[i]
                            for i in env.action_history
                        ],
                    )
                ),

                # SQL realmente submetida ao EXPLAIN ao final.
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

                "erro": info_final.get(
                    "erro",
                    "",
                ),
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
    print("BASELINE RANDOM FINALIZADO")
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
