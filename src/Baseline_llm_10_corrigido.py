import csv
import json
import random
import re
from datetime import datetime

import numpy as np
import requests

from LearningJob_v4_01_corrigido import LearningJob


# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
EPISODIOS = 500
SEED = 42

ARQUIVO_SAIDA = "resultados_llm_baseline.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO_LLM = "qwen2.5-coder:7b"
TIMEOUT_OLLAMA = 180


# =====================================================================
# REPRODUTIBILIDADE
# =====================================================================
def configurar_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


# =====================================================================
# CHAMADA AO QWEN
# =====================================================================
def chamar_qwen(query, aliases, joins_validos):
    """
    Solicita ao Qwen uma sequência de pares de JOIN.

    A temperatura é fixada em 0 para reduzir variabilidade entre chamadas e
    tornar o baseline mais reprodutível.
    """
    numero_joins_necessarios = max(len(aliases) - 1, 0)

    prompt = f"""
Você é um otimizador SQL especializado exclusivamente em escolher a ordem
das operações INNER JOIN.

Consulta SQL:
{query}

Aliases e tabelas:
{aliases}

Pares de JOIN relacionais válidos:
{joins_validos}

Sua tarefa é escolher uma sequência de JOINs que conecte todas as tabelas.

REGRAS OBRIGATÓRIAS:
1. Use SOMENTE aliases existentes na consulta.
2. Use SOMENTE pares presentes na lista de JOINs válidos.
3. Não crie CROSS JOIN.
4. Não repita pares.
5. A sequência deve conectar todos os aliases.
6. Para esta consulta, forneça até {numero_joins_necessarios} operações de JOIN.
7. Responda SOMENTE com JSON válido, sem Markdown e sem explicações.

Formato obrigatório:
{{
  "join_order": [
    ["alias1", "alias2"],
    ["alias2", "alias3"]
  ]
}}
""".strip()

    resposta = requests.post(
        OLLAMA_URL,
        json={
            "model": MODELO_LLM,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "seed": SEED,
            },
        },
        timeout=TIMEOUT_OLLAMA,
    )

    resposta.raise_for_status()

    payload = resposta.json()
    return payload.get("response", "")


# =====================================================================
# EXTRAÇÃO DA RESPOSTA JSON
# =====================================================================
def extrair_json_resposta(texto):
    """
    Tenta interpretar diretamente o JSON e, se necessário, extrai o primeiro
    objeto JSON encontrado na resposta.
    """
    if not texto:
        return {"join_order": []}

    try:
        dados = json.loads(texto)
        if isinstance(dados, dict):
            return dados
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", str(texto), re.DOTALL)

    if not match:
        return {"join_order": []}

    try:
        dados = json.loads(match.group(0))
        if isinstance(dados, dict):
            return dados
    except (json.JSONDecodeError, TypeError):
        pass

    return {"join_order": []}


# =====================================================================
# CONVERSÃO DA ORDEM DA LLM PARA AÇÕES DO AMBIENTE
# =====================================================================
def converter_ordem_em_acoes(env, join_order):
    """
    Converte os pares sugeridos pela LLM em índices de ação do ambiente.
    Remove pares inexistentes e duplicações.
    """
    acoes = []
    vistos = set()

    if not isinstance(join_order, list):
        return acoes

    for par in join_order:
        if (
            not isinstance(par, (list, tuple))
            or len(par) != 2
            or not all(isinstance(x, str) for x in par)
        ):
            continue

        par_normalizado = tuple(sorted((par[0], par[1])))

        if par_normalizado in vistos:
            continue

        for idx, pair in enumerate(env.action_pairs):
            if tuple(sorted(pair)) == par_normalizado:
                acoes.append(idx)
                vistos.add(par_normalizado)
                break

    return acoes


# =====================================================================
# FALLBACK
# =====================================================================
def escolher_acao_fallback(env):
    """
    Caso a LLM não forneça uma ação utilizável para o estado atual,
    seleciona deterministicamente a primeira ação válida disponível.
    """
    validas = env.valid_action_indices()

    if not validas:
        return None

    return min(validas)


# =====================================================================
# EXECUÇÃO DO BASELINE LLM
# =====================================================================
def executar_baseline_llm():
    configurar_seed(SEED)

    print("=" * 70)
    print("BASELINE LLM - QWEN2.5-CODER - ORDENAÇÃO DE JOINS")
    print("=" * 70)
    print(f"Episódios: {EPISODIOS}")
    print(f"Seed: {SEED}")
    print(f"Modelo: {MODELO_LLM}")
    print("Temperatura: 0")

    env = LearningJob()
    resultados = []

    try:
        for episodio in range(1, EPISODIOS + 1):

            # Inicializa o RNG do Gymnasium apenas no primeiro episódio.
            if episodio == 1:
                state, info_reset = env.reset(seed=SEED)
            else:
                state, info_reset = env.reset()

            resposta_llm = ""
            erro_llm = ""
            join_order = []

            try:
                resposta_llm = chamar_qwen(
                    query=env.consulta_atual,
                    aliases=info_reset.get("aliases", {}),
                    joins_validos=info_reset.get("joins_validos", []),
                )

                dados_llm = extrair_json_resposta(resposta_llm)
                join_order = dados_llm.get("join_order", [])

            except Exception as exc:
                # Uma falha pontual do Ollama não aborta todo o experimento.
                erro_llm = f"{type(exc).__name__}: {exc}"
                join_order = []

            acoes_llm = converter_ordem_em_acoes(
                env,
                join_order,
            )

            ponteiro_acao = 0

            terminated = False
            truncated = False

            recompensa_total = 0.0
            passos = 0

            info_final = {}

            usou_fallback = False
            qtd_fallback = 0
            qtd_acoes_llm_usadas = 0

            while not terminated and not truncated:
                action = None

                # Primeiro tenta seguir a ordem proposta pela LLM.
                while ponteiro_acao < len(acoes_llm):
                    candidata = acoes_llm[ponteiro_acao]
                    ponteiro_acao += 1

                    if candidata not in env.valid_action_indices():
                        continue

                    action = candidata
                    qtd_acoes_llm_usadas += 1
                    break

                # Se a proposta acabou ou ficou inválida após as fusões
                # anteriores, usa fallback.
                if action is None:
                    action = escolher_acao_fallback(env)

                    if action is not None:
                        usou_fallback = True
                        qtd_fallback += 1

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
                "data_execucao": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "modelo_llm": MODELO_LLM,

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

                "resposta_llm": resposta_llm,
                "ordem_llm_extraida": str(join_order),
                "acoes_llm_convertidas": str(acoes_llm),

                "qtd_acoes_llm_usadas": qtd_acoes_llm_usadas,
                "qtd_fallback": qtd_fallback,
                "usou_fallback": usou_fallback,

                # Ordem REAL de execução.
                "ordem_logica_joins": str(
                    info_final.get(
                        "ordem_logica_joins",
                        [
                            env.action_pairs[i]
                            for i in env.action_history
                        ],
                    )
                ),

                # SQL efetivamente submetida ao EXPLAIN.
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

                "erro_llm": erro_llm,
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
                f"fallback={usou_fallback} "
                f"({qtd_fallback}) | "
                f"ações_llm={qtd_acoes_llm_usadas} | "
                f"reward={recompensa_total:.6f} | "
                f"custo={custo_texto} | "
                f"melhoria={melhoria_texto}"
            )

            if erro_llm:
                print(
                    f"  Aviso LLM: {erro_llm}"
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

    # =================================================================
    # RESUMO FINAL
    # =================================================================
    finalizados = sum(
        bool(r["finalizado"])
        for r in resultados
    )

    episodios_com_fallback = sum(
        bool(r["usou_fallback"])
        for r in resultados
    )

    total_acoes_fallback = sum(
        int(r["qtd_fallback"])
        for r in resultados
    )

    erros_llm = sum(
        bool(r["erro_llm"])
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

    print("\n" + "=" * 70)
    print("BASELINE LLM FINALIZADO")
    print("=" * 70)

    print(f"CSV salvo: {ARQUIVO_SAIDA}")
    print(
        f"Episódios finalizados: "
        f"{finalizados}/{len(resultados)}"
    )
    print(
        f"Episódios com fallback: "
        f"{episodios_com_fallback}/{len(resultados)} "
        f"({100.0 * episodios_com_fallback / len(resultados):.2f}%)"
    )
    print(
        f"Total de ações por fallback: "
        f"{total_acoes_fallback}"
    )
    print(
        f"Erros de chamada ao LLM: "
        f"{erros_llm}"
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
    executar_baseline_llm()
