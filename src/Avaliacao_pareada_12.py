import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn

from LearningJob_v4_01_corrigido import LearningJob


# =====================================================================
# CONFIGURAÇÕES GERAIS
# =====================================================================
SEED = 42
RANDOM_REPS_DEFAULT = 30

ARQUIVO_MODELO_DQN = "modelo_dqn_join_order.pth"
ARQUIVO_MODELO_DDQN = "modelo_ddqn_join_order.pth"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO_LLM = "qwen2.5-coder:7b"
TIMEOUT_OLLAMA = 180

TOLERANCIA_EMPATE = 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# REPRODUTIBILIDADE
# =====================================================================
def configurar_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# REDE DQN/DDQN - MESMA ARQUITETURA USADA NO TREINAMENTO
# =====================================================================
class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


def carregar_modelo(caminho: str, env: LearningJob) -> DQN:
    path = Path(caminho)
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path.resolve()}\n"
            "Coloque o arquivo .pth na mesma pasta da avaliação ou informe o caminho correto."
        )

    checkpoint = torch.load(path, map_location=DEVICE)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    ck_state = int(checkpoint.get("state_size", state_size))
    ck_action = int(checkpoint.get("action_size", action_size))

    if ck_state != state_size or ck_action != action_size:
        raise ValueError(
            f"Dimensões incompatíveis em {path.name}: "
            f"checkpoint=({ck_state}, {ck_action}), ambiente=({state_size}, {action_size})."
        )

    model = DQN(state_size, action_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def escolher_acao_modelo(model: DQN, state: np.ndarray, env: LearningJob) -> Optional[int]:
    validas = env.valid_action_indices()
    if not validas:
        return None

    x = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        q_values = model(x).squeeze(0)

    mascara = torch.full_like(q_values, float("-inf"))
    mascara[validas] = q_values[validas]
    return int(torch.argmax(mascara).item())


# =====================================================================
# GREEDY - MESMA HEURÍSTICA DO BASELINE CORRIGIDO
# =====================================================================
def ainda_tem_caminho_para_finalizar(env: LearningJob, action_idx: int) -> bool:
    componentes = [set(c) for c in env.components]
    left_alias, right_alias = env.action_pairs[action_idx]

    try:
        comp_left_idx = env._find_component_index(left_alias)
        comp_right_idx = env._find_component_index(right_alias)
    except ValueError:
        return False

    if comp_left_idx == comp_right_idx:
        return False

    novo_componente = componentes[comp_left_idx] | componentes[comp_right_idx]

    for idx in sorted([comp_left_idx, comp_right_idx], reverse=True):
        del componentes[idx]
    componentes.append(novo_componente)

    if len(componentes) == 1:
        return True

    grafo_componentes = {i: set() for i in range(len(componentes))}

    for a, b in env.join_edges:
        comp_a = None
        comp_b = None

        for idx, comp in enumerate(componentes):
            if a in comp:
                comp_a = idx
            if b in comp:
                comp_b = idx

        if comp_a is not None and comp_b is not None and comp_a != comp_b:
            grafo_componentes[comp_a].add(comp_b)
            grafo_componentes[comp_b].add(comp_a)

    visitados = set()
    pilha = [0]

    while pilha:
        atual = pilha.pop()
        if atual in visitados:
            continue
        visitados.add(atual)
        pilha.extend(grafo_componentes[atual] - visitados)

    return len(visitados) == len(componentes)


def escolher_acao_greedy(env: LearningJob) -> Optional[int]:
    candidatas = []

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

        # Maior score; empate pelo menor índice de ação.
        candidatas.append((score, -idx, idx))

    if not candidatas:
        return None

    candidatas.sort(reverse=True)
    return candidatas[0][2]


# =====================================================================
# QWEN / OLLAMA
# =====================================================================
def chamar_qwen(query, aliases, joins_validos):
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


def extrair_json_resposta(texto):
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


def converter_ordem_em_acoes(env: LearningJob, join_order):
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


def escolher_acao_fallback(env: LearningJob) -> Optional[int]:
    validas = env.valid_action_indices()
    return min(validas) if validas else None


# =====================================================================
# UTILIDADES
# =====================================================================
def melhoria_percentual(custo_original, custo_final) -> Optional[float]:
    if (
        custo_original is None
        or custo_final is None
        or not np.isfinite(custo_original)
        or not np.isfinite(custo_final)
        or custo_original == 0
    ):
        return None

    return (custo_original - custo_final) / custo_original * 100.0


def serializar_ordem(ordem) -> str:
    return json.dumps(ordem or [], ensure_ascii=False)


def hash_consulta(sql: str) -> str:
    return hashlib.sha1(sql.encode("utf-8")).hexdigest()[:12]


def construir_mapa_nomes_consultas(env: LearningJob) -> Dict[str, str]:
    """Mapeia a SQL limpa ao nome consultaXX.sql, quando possível."""
    mapa = {}
    diretorio = Path(env.diretorio_consultas)

    if not diretorio.exists():
        return mapa

    for i in range(1, 30):
        nome = f"consulta{i:02d}.sql"
        path = diretorio / nome
        if not path.exists():
            continue

        try:
            raw = path.read_text(encoding="utf-8")
            limpa = env._clean_sql(raw)
            mapa[limpa] = nome
        except Exception:
            pass

    return mapa


def executar_episodio_com_politica(
    env: LearningJob,
    query_index: int,
    politica,
    seed_reset: Optional[int] = None,
):
    state, info_reset = env.reset(
        seed=seed_reset,
        options={"query_index": query_index},
    )

    terminated = False
    truncated = False
    recompensa_total = 0.0
    passos = 0
    info_final = {}

    while not terminated and not truncated:
        action = politica(state, env)

        if action is None:
            truncated = True
            recompensa_total -= 1000.0
            info_final = {
                "erro": "Nenhuma ação válida restante",
                "custo_estimado_final": float("inf"),
                "cardinalidade_estimada_final": float("inf"),
                "consulta_sql_reordenada": "",
                "ordem_logica_joins": [
                    env.action_pairs[i] for i in env.action_history
                ],
            }
            break

        state, reward, terminated, truncated, step_info = env.step(action)
        recompensa_total += float(reward)
        passos += 1

        if step_info:
            info_final.update(step_info)

    return info_reset, info_final, recompensa_total, passos, terminated, truncated


def linha_resultado_base(
    cenario: str,
    metodo: str,
    query_index: int,
    query_name: str,
    info_reset: dict,
    info_final: dict,
    recompensa_total: float,
    passos: int,
    terminated: bool,
    truncated: bool,
    repeticao: Optional[int] = None,
    seed_random: Optional[int] = None,
    extras: Optional[dict] = None,
):
    sql_original = info_reset.get("consulta_selecionada", "")
    custo_original = info_reset.get("custo_postgresql")
    custo_final = info_final.get("custo_estimado_final")

    row = {
        "cenario": cenario,
        "metodo": metodo,
        "query_index": query_index,
        "consulta_nome": query_name,
        "consulta_hash": hash_consulta(sql_original),
        "qtd_tabelas": len(info_reset.get("aliases", {})),
        "repeticao": repeticao,
        "seed_random": seed_random,
        "custo_original": custo_original,
        "custo_final": custo_final,
        "melhoria_percentual": melhoria_percentual(custo_original, custo_final),
        "cardinalidade_original": info_reset.get("cardinalidade_postgresql"),
        "cardinalidade_final": info_final.get("cardinalidade_estimada_final"),
        "recompensa_total": recompensa_total,
        "passos": passos,
        "finalizado": bool(terminated and info_final.get("episodio_finalizado", True)),
        "truncado": bool(truncated),
        "ordem_logica_joins": serializar_ordem(info_final.get("ordem_logica_joins", [])),
        "consulta_sql_original": sql_original,
        "consulta_sql_reordenada": info_final.get("consulta_sql_reordenada", ""),
        "erro": info_final.get("erro", ""),
        "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if extras:
        row.update(extras)

    return row


# =====================================================================
# EXECUÇÃO DE CADA MÉTODO
# =====================================================================
def avaliar_random(
    env: LearningJob,
    query_index: int,
    query_name: str,
    cenario: str,
    random_reps: int,
) -> List[dict]:
    rows = []

    for rep in range(1, random_reps + 1):
        # Determinístico por consulta/repetição e reutilizável entre cenários.
        seed_rep = SEED * 1_000_000 + query_index * 10_000 + rep
        rng = np.random.default_rng(seed_rep)

        def politica(_state, e):
            validas = e.valid_action_indices()
            if not validas:
                return None
            return int(rng.choice(validas))

        result = executar_episodio_com_politica(
            env,
            query_index,
            politica,
        )

        rows.append(
            linha_resultado_base(
                cenario,
                "Random",
                query_index,
                query_name,
                *result,
                repeticao=rep,
                seed_random=seed_rep,
            )
        )

    return rows


def avaliar_greedy(
    env: LearningJob,
    query_index: int,
    query_name: str,
    cenario: str,
) -> dict:
    def politica(_state, e):
        return escolher_acao_greedy(e)

    result = executar_episodio_com_politica(
        env,
        query_index,
        politica,
    )

    return linha_resultado_base(
        cenario,
        "Greedy",
        query_index,
        query_name,
        *result,
    )


def avaliar_modelo(
    env: LearningJob,
    model: DQN,
    nome_metodo: str,
    query_index: int,
    query_name: str,
    cenario: str,
) -> dict:
    def politica(state, e):
        return escolher_acao_modelo(model, state, e)

    result = executar_episodio_com_politica(
        env,
        query_index,
        politica,
    )

    return linha_resultado_base(
        cenario,
        nome_metodo,
        query_index,
        query_name,
        *result,
    )


def avaliar_qwen(
    env: LearningJob,
    query_index: int,
    query_name: str,
    cenario: str,
) -> dict:
    state, info_reset = env.reset(options={"query_index": query_index})

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
        erro_llm = f"{type(exc).__name__}: {exc}"
        join_order = []

    acoes_llm = converter_ordem_em_acoes(env, join_order)
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

        while ponteiro_acao < len(acoes_llm):
            candidata = acoes_llm[ponteiro_acao]
            ponteiro_acao += 1

            if candidata not in env.valid_action_indices():
                continue

            action = candidata
            qtd_acoes_llm_usadas += 1
            break

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
                    env.action_pairs[i] for i in env.action_history
                ],
            }
            break

        state, reward, terminated, truncated, step_info = env.step(action)
        recompensa_total += float(reward)
        passos += 1

        if step_info:
            info_final.update(step_info)

    extras = {
        "modelo_llm": MODELO_LLM,
        "usou_fallback": usou_fallback,
        "qtd_fallback": qtd_fallback,
        "qtd_acoes_llm_usadas": qtd_acoes_llm_usadas,
        "erro_llm": erro_llm,
        "resposta_llm": resposta_llm,
        "ordem_llm_extraida": json.dumps(join_order, ensure_ascii=False),
        "acoes_llm_convertidas": json.dumps(acoes_llm),
    }

    return linha_resultado_base(
        cenario,
        "Qwen",
        query_index,
        query_name,
        info_reset,
        info_final,
        recompensa_total,
        passos,
        terminated,
        truncated,
        extras=extras,
    )


# =====================================================================
# CONSOLIDAÇÃO PAREADA
# =====================================================================
def consolidar_por_consulta(rows: List[dict]) -> List[dict]:
    grupos = defaultdict(list)

    for row in rows:
        if row.get("finalizado") and row.get("custo_final") is not None:
            custo = row.get("custo_final")
            if custo is not None and np.isfinite(float(custo)):
                grupos[(row["metodo"], row["query_index"])].append(row)

    saida = []

    for (metodo, query_index), itens in sorted(grupos.items(), key=lambda x: (x[0][1], x[0][0])):
        custos = np.array([float(x["custo_final"]) for x in itens], dtype=float)
        cards = np.array([float(x["cardinalidade_final"]) for x in itens], dtype=float)
        custo_original = float(itens[0]["custo_original"])
        custo_rep = float(np.mean(custos))

        saida.append({
            "cenario": itens[0]["cenario"],
            "metodo": metodo,
            "query_index": query_index,
            "consulta_nome": itens[0]["consulta_nome"],
            "consulta_hash": itens[0]["consulta_hash"],
            "qtd_tabelas": itens[0]["qtd_tabelas"],
            "n_repeticoes": len(itens),
            "custo_original": custo_original,
            "custo_metodo": custo_rep,
            "custo_min": float(np.min(custos)),
            "custo_max": float(np.max(custos)),
            "custo_std": float(np.std(custos, ddof=1)) if len(custos) > 1 else 0.0,
            "cardinalidade_media": float(np.mean(cards)),
            "melhoria_percentual": melhoria_percentual(custo_original, custo_rep),
            "usou_fallback": any(bool(x.get("usou_fallback", False)) for x in itens),
            "qtd_fallback": int(sum(int(x.get("qtd_fallback", 0) or 0) for x in itens)),
        })

    return saida


def resumir_metodos(consolidado: List[dict], somente_3_ou_mais: bool = False) -> List[dict]:
    dados = [
        x for x in consolidado
        if (not somente_3_ou_mais or int(x["qtd_tabelas"]) >= 3)
    ]

    grupos = defaultdict(list)
    for row in dados:
        grupos[row["metodo"]].append(row)

    ordem_metodos = ["Random", "Greedy", "DQN", "DDQN", "Qwen"]
    saida = []

    for metodo in ordem_metodos:
        itens = grupos.get(metodo, [])
        if not itens:
            continue

        custos_orig = np.array([float(x["custo_original"]) for x in itens])
        custos_met = np.array([float(x["custo_metodo"]) for x in itens])
        melhorias = np.array([float(x["melhoria_percentual"]) for x in itens])

        ganhos = int(np.sum(custos_met < custos_orig - TOLERANCIA_EMPATE))
        empates = int(np.sum(np.isclose(custos_met, custos_orig, rtol=0, atol=TOLERANCIA_EMPATE)))
        perdas = int(len(itens) - ganhos - empates)

        melhoria_global = (
            (float(np.sum(custos_orig)) - float(np.sum(custos_met)))
            / float(np.sum(custos_orig))
            * 100.0
        ) if float(np.sum(custos_orig)) != 0 else None

        saida.append({
            "escopo": "3+ tabelas" if somente_3_ou_mais else "todas as consultas",
            "metodo": metodo,
            "n_consultas": len(itens),
            "custo_medio": float(np.mean(custos_met)),
            "custo_mediano": float(np.median(custos_met)),
            "soma_custo_original": float(np.sum(custos_orig)),
            "soma_custo_metodo": float(np.sum(custos_met)),
            "melhoria_global_percentual": melhoria_global,
            "melhoria_media_por_consulta_percentual": float(np.mean(melhorias)),
            "melhoria_mediana_por_consulta_percentual": float(np.median(melhorias)),
            "desvio_padrao_melhoria_percentual": float(np.std(melhorias, ddof=1)) if len(melhorias) > 1 else 0.0,
            "vitorias": ganhos,
            "empates": empates,
            "derrotas": perdas,
            "taxa_vitoria_percentual": ganhos / len(itens) * 100.0,
            "consultas_com_fallback": int(sum(bool(x.get("usou_fallback", False)) for x in itens)),
            "acoes_fallback": int(sum(int(x.get("qtd_fallback", 0) or 0) for x in itens)),
        })

    return saida


def validar_pareamento(consolidado: List[dict]):
    por_metodo = defaultdict(set)
    for row in consolidado:
        por_metodo[row["metodo"]].add(row["query_index"])

    conjuntos = list(por_metodo.values())
    if not conjuntos:
        raise RuntimeError("Nenhum resultado consolidado foi produzido.")

    referencia = conjuntos[0]
    divergentes = {
        metodo: sorted(ids)
        for metodo, ids in por_metodo.items()
        if ids != referencia
    }

    if divergentes:
        print("\nATENÇÃO: o pareamento não ficou completo para todos os métodos:")
        for metodo, ids in divergentes.items():
            print(f"  {metodo}: {ids}")
    else:
        print(f"Pareamento validado: todos os métodos possuem as mesmas {len(referencia)} consultas.")


# =====================================================================
# CSV
# =====================================================================
def salvar_csv(caminho: Path, rows: List[dict]):
    if not rows:
        return

    campos = []
    vistos = set()
    for row in rows:
        for key in row.keys():
            if key not in vistos:
                campos.append(key)
                vistos.add(key)

    with caminho.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
def executar_avaliacao(
    cenario: str,
    random_reps: int,
    modelo_dqn: str,
    modelo_ddqn: str,
    pular_llm: bool = False,
):
    configurar_seed(SEED)

    print("=" * 78)
    print("AVALIAÇÃO PAREADA - RANDOM / GREEDY / DQN / DDQN / QWEN")
    print("=" * 78)
    print(f"Cenário: {cenario}")
    print(f"Seed base: {SEED}")
    print(f"Random: {random_reps} repetições por consulta")
    print(f"Dispositivo DQN/DDQN: {DEVICE}")
    print(f"Qwen: {'DESATIVADO' if pular_llm else MODELO_LLM + ' | temperature=0'}")
    print()

    env = LearningJob()
    nomes_consultas = construir_mapa_nomes_consultas(env)

    dqn = carregar_modelo(modelo_dqn, env)
    ddqn = carregar_modelo(modelo_ddqn, env)

    print(f"Consultas válidas carregadas: {len(env.consultas_sql)}")
    print(f"DQN carregado: {modelo_dqn}")
    print(f"DDQN carregado: {modelo_ddqn}")

    rows_detalhados = []

    try:
        for query_index, sql in enumerate(env.consultas_sql):
            query_name = nomes_consultas.get(sql, f"query_index_{query_index:02d}")

            # Reset rápido apenas para exibir metadados da consulta.
            _, meta = env.reset(options={"query_index": query_index})
            n_tables = len(meta.get("aliases", {}))
            custo_orig = meta.get("custo_postgresql")

            print("\n" + "-" * 78)
            print(
                f"[{query_index + 1:02d}/{len(env.consultas_sql):02d}] "
                f"{query_name} | tabelas={n_tables} | custo original={custo_orig:.2f}"
            )

            # Random: várias repetições, mesmas seeds entre cenários.
            random_rows = avaliar_random(
                env,
                query_index,
                query_name,
                cenario,
                random_reps,
            )
            rows_detalhados.extend(random_rows)
            random_mean = np.mean([float(x["custo_final"]) for x in random_rows])
            print(
                f"  Random | custo médio={random_mean:.2f} | "
                f"melhoria={melhoria_percentual(custo_orig, random_mean):+.4f}%"
            )

            # Greedy
            row = avaliar_greedy(env, query_index, query_name, cenario)
            rows_detalhados.append(row)
            print(
                f"  Greedy | custo={float(row['custo_final']):.2f} | "
                f"melhoria={float(row['melhoria_percentual']):+.4f}%"
            )

            # DQN - política congelada / epsilon = 0
            row = avaliar_modelo(env, dqn, "DQN", query_index, query_name, cenario)
            rows_detalhados.append(row)
            print(
                f"  DQN    | custo={float(row['custo_final']):.2f} | "
                f"melhoria={float(row['melhoria_percentual']):+.4f}%"
            )

            # DDQN - política congelada / epsilon = 0
            row = avaliar_modelo(env, ddqn, "DDQN", query_index, query_name, cenario)
            rows_detalhados.append(row)
            print(
                f"  DDQN   | custo={float(row['custo_final']):.2f} | "
                f"melhoria={float(row['melhoria_percentual']):+.4f}%"
            )

            # Qwen: uma chamada por consulta, temperature 0.
            if not pular_llm:
                row = avaliar_qwen(env, query_index, query_name, cenario)
                rows_detalhados.append(row)
                print(
                    f"  Qwen   | custo={float(row['custo_final']):.2f} | "
                    f"melhoria={float(row['melhoria_percentual']):+.4f}% | "
                    f"fallback={row['usou_fallback']} ({row['qtd_fallback']})"
                )

    finally:
        env.close()

    consolidado = consolidar_por_consulta(rows_detalhados)
    validar_pareamento(consolidado)

    resumo_todas = resumir_metodos(consolidado, somente_3_ou_mais=False)
    resumo_3mais = resumir_metodos(consolidado, somente_3_ou_mais=True)
    resumo = resumo_todas + resumo_3mais

    prefixo = f"avaliacao_pareada_{cenario}"
    arquivo_detalhado = Path(f"{prefixo}_detalhado.csv")
    arquivo_consulta = Path(f"{prefixo}_por_consulta.csv")
    arquivo_resumo = Path(f"{prefixo}_resumo.csv")

    salvar_csv(arquivo_detalhado, rows_detalhados)
    salvar_csv(arquivo_consulta, consolidado)
    salvar_csv(arquivo_resumo, resumo)

    print("\n" + "=" * 78)
    print("AVALIAÇÃO PAREADA FINALIZADA")
    print("=" * 78)
    print(f"Detalhado:    {arquivo_detalhado}")
    print(f"Por consulta: {arquivo_consulta}")
    print(f"Resumo:        {arquivo_resumo}")

    print("\nRESUMO PRINCIPAL - TODAS AS CONSULTAS")
    print("-" * 78)
    print(
        f"{'Método':<10} {'N':>3} {'Custo médio':>15} {'Melh.global':>13} "
        f"{'Mediana %':>11} {'V/E/D':>11}"
    )

    for row in resumo_todas:
        print(
            f"{row['metodo']:<10} "
            f"{row['n_consultas']:>3d} "
            f"{row['custo_medio']:>15.2f} "
            f"{row['melhoria_global_percentual']:>+12.4f}% "
            f"{row['melhoria_mediana_por_consulta_percentual']:>+10.4f}% "
            f"{row['vitorias']:>2d}/{row['empates']:>2d}/{row['derrotas']:>2d}"
        )

    print("\nObservação: 'Melh.global' usa (Σ custo_original - Σ custo_método) / Σ custo_original,")
    print("evitando que percentuais extremos de consultas de custo muito baixo dominem a média.")


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Avaliação pareada dos métodos de ordenação de JOINs."
    )
    parser.add_argument(
        "--cenario",
        default="original",
        choices=["original", "indexado"],
        help="Rótulo do cenário atual do banco. O script não cria/remove índices.",
    )
    parser.add_argument(
        "--random-reps",
        type=int,
        default=RANDOM_REPS_DEFAULT,
        help="Número de repetições do Random por consulta (padrão: 30).",
    )
    parser.add_argument(
        "--modelo-dqn",
        default=ARQUIVO_MODELO_DQN,
        help="Caminho do checkpoint DQN corrigido.",
    )
    parser.add_argument(
        "--modelo-ddqn",
        default=ARQUIVO_MODELO_DDQN,
        help="Caminho do checkpoint DDQN corrigido.",
    )
    parser.add_argument(
        "--pular-llm",
        action="store_true",
        help="Executa avaliação sem Qwen, útil para diagnóstico rápido.",
    )

    args = parser.parse_args()

    if args.random_reps < 1:
        raise ValueError("--random-reps deve ser >= 1")

    executar_avaliacao(
        cenario=args.cenario,
        random_reps=args.random_reps,
        modelo_dqn=args.modelo_dqn,
        modelo_ddqn=args.modelo_ddqn,
        pular_llm=args.pular_llm,
    )


if __name__ == "__main__":
    main()
