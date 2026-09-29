import csv
import random
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Se você salvar o ambiente corrigido com outro nome, ajuste este import.
from LearningJob_v4_01_corrigido import LearningJob


# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
EPISODIOS = 500
BATCH_SIZE = 64
GAMMA = 0.95
LR = 0.001

EPSILON_INICIAL = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY = 0.995

MEMORIA_MAX = 10000
TARGET_UPDATE_EPISODIOS = 20

SEED = 42

ARQUIVO_SAIDA = "resultados_ddqn.csv"
ARQUIVO_MODELO = "modelo_ddqn_join_order.pth"


# =====================================================================
# DISPOSITIVO
# =====================================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# REDE DDQN
# =====================================================================
class DDQN(nn.Module):
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


# =====================================================================
# MÁSCARA DE AÇÕES VÁLIDAS
# =====================================================================
def criar_mascara_acoes_validas(
    env: LearningJob,
    action_size: int
) -> np.ndarray:
    """
    Cria uma máscara booleana para as ações válidas no estado atual.

    True  -> ação válida
    False -> ação inválida
    """
    mask = np.zeros(action_size, dtype=np.bool_)

    for action in env.valid_action_indices():
        if 0 <= action < action_size:
            mask[action] = True

    return mask


# =====================================================================
# ESCOLHA DE AÇÃO
# =====================================================================
def escolher_acao(
    model: DDQN,
    state: np.ndarray,
    env: LearningJob,
    epsilon: float,
) -> Optional[int]:
    """
    Política epsilon-greedy considerando somente ações válidas.
    """
    acoes_validas = env.valid_action_indices()

    if not acoes_validas:
        return None

    if random.random() < epsilon:
        return random.choice(acoes_validas)

    state_tensor = (
        torch.as_tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        )
        .unsqueeze(0)
    )

    model.eval()

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)

    model.train()

    valid_indices = torch.tensor(
        acoes_validas,
        dtype=torch.long,
        device=DEVICE,
    )

    q_validos = q_values[valid_indices]
    melhor_posicao = torch.argmax(q_validos).item()

    return acoes_validas[melhor_posicao]


# =====================================================================
# REPLAY BUFFER / TREINAMENTO DDQN
# =====================================================================
def treinar_replay(
    model: DDQN,
    target_model: DDQN,
    memoria: deque,
    optimizer: optim.Optimizer,
):
    """
    Atualização Double DQN com máscara das ações válidas.

    Diferença central em relação ao DQN:

      1. A rede ONLINE (model) escolhe a melhor ação no próximo estado.
      2. A rede ALVO (target_model) avalia o valor dessa ação.

    Além disso, ações inválidas são mascaradas antes da escolha da ação
    seguinte, evitando que elas influenciem o alvo de treinamento.

    Cada transição contém:
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_mask
    """
    if len(memoria) < BATCH_SIZE:
        return None

    batch = random.sample(memoria, BATCH_SIZE)

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_valid_masks,
    ) = zip(*batch)

    states_tensor = torch.as_tensor(
        np.asarray(states),
        dtype=torch.float32,
        device=DEVICE,
    )

    actions_tensor = torch.as_tensor(
        actions,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(1)

    rewards_tensor = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(1)

    next_states_tensor = torch.as_tensor(
        np.asarray(next_states),
        dtype=torch.float32,
        device=DEVICE,
    )

    dones_tensor = torch.as_tensor(
        dones,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(1)

    next_valid_masks_tensor = torch.as_tensor(
        np.asarray(next_valid_masks),
        dtype=torch.bool,
        device=DEVICE,
    )

    # Q(s,a) da ação efetivamente executada.
    q_values = model(states_tensor).gather(
        1,
        actions_tensor,
    )

    with torch.no_grad():
        # -------------------------------------------------------------
        # DOUBLE DQN
        # -------------------------------------------------------------
        # A rede online SELECIONA a melhor ação para o próximo estado.
        online_next_q_all = model(next_states_tensor)

        masked_online_next_q = online_next_q_all.masked_fill(
            ~next_valid_masks_tensor,
            float("-inf"),
        )

        has_valid_action = next_valid_masks_tensor.any(
            dim=1,
            keepdim=True,
        )

        next_actions = masked_online_next_q.argmax(
            dim=1,
            keepdim=True,
        )

        # A rede target AVALIA a ação escolhida pela rede online.
        target_next_q_all = target_model(next_states_tensor)

        next_q_values = target_next_q_all.gather(
            1,
            next_actions,
        )

        # Estado terminal ou sem ações válidas -> valor futuro = 0.
        next_q_values = torch.where(
            has_valid_action,
            next_q_values,
            torch.zeros_like(next_q_values),
        )

        target = (
            rewards_tensor
            + GAMMA * next_q_values * (1.0 - dones_tensor)
        )

    loss = nn.MSELoss()(q_values, target)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    return float(loss.item())


# =====================================================================
# EXECUÇÃO DO TREINAMENTO
# =====================================================================
def executar_ddqn():
    configurar_seed(SEED)

    print("=" * 70)
    print("TREINAMENTO DDQN - ORDENAÇÃO DE JOINS")
    print("=" * 70)
    print(f"Dispositivo: {DEVICE}")
    print(f"Episódios: {EPISODIOS}")
    print(f"Seed: {SEED}")

    env = LearningJob()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DDQN(
        state_size,
        action_size,
    ).to(DEVICE)

    target_model = DDQN(
        state_size,
        action_size,
    ).to(DEVICE)

    target_model.load_state_dict(
        model.state_dict()
    )

    target_model.eval()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    memoria = deque(
        maxlen=MEMORIA_MAX
    )

    epsilon = EPSILON_INICIAL
    resultados = []

    try:
        for episodio in range(1, EPISODIOS + 1):

            # Durante o TREINAMENTO, a consulta continua sendo sorteada.
            state, info_reset = env.reset()

            terminated = False
            truncated = False

            recompensa_total = 0.0
            passos = 0

            info_final = {}
            perdas_episodio: List[float] = []

            while not terminated and not truncated:
                action = escolher_acao(
                    model=model,
                    state=state,
                    env=env,
                    epsilon=epsilon,
                )

                # Caso excepcional: nenhum caminho de JOIN válido restante.
                if action is None:
                    reward = -1000.0

                    terminated = False
                    truncated = True

                    recompensa_total += reward

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

                    # Não armazenamos uma transição artificial com action=0.
                    break

                next_state, reward, terminated, truncated, step_info = env.step(
                    action
                )

                done = terminated or truncated

                # A máscara armazenada deve representar o ESTADO SEGUINTE.
                if done:
                    next_valid_mask = np.zeros(
                        action_size,
                        dtype=np.bool_,
                    )
                else:
                    next_valid_mask = criar_mascara_acoes_validas(
                        env,
                        action_size,
                    )

                memoria.append(
                    (
                        state.copy(),
                        int(action),
                        float(reward),
                        next_state.copy(),
                        bool(done),
                        next_valid_mask.copy(),
                    )
                )

                state = next_state
                recompensa_total += reward
                passos += 1

                if step_info:
                    info_final.update(step_info)

                loss = treinar_replay(
                    model=model,
                    target_model=target_model,
                    memoria=memoria,
                    optimizer=optimizer,
                )

                if loss is not None:
                    perdas_episodio.append(loss)

            # Epsilon decai uma vez por episódio.
            epsilon = max(
                EPSILON_FINAL,
                epsilon * EPSILON_DECAY,
            )

            # Atualização periódica da target network.
            if episodio % TARGET_UPDATE_EPISODIOS == 0:
                target_model.load_state_dict(
                    model.state_dict()
                )

            loss_medio = (
                float(np.mean(perdas_episodio))
                if perdas_episodio
                else None
            )

            custo_original = info_reset.get(
                "custo_postgresql"
            )

            custo_final = info_final.get(
                "custo_estimado_final"
            )

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

                # Ordem real em que as ações foram executadas.
                "ordem_logica_joins": str(
                    info_final.get(
                        "ordem_logica_joins",
                        [
                            env.action_pairs[i]
                            for i in env.action_history
                        ],
                    )
                ),

                # SQL que efetivamente recebeu EXPLAIN no final.
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
                "epsilon": epsilon,
                "loss_medio": loss_medio,

                "finalizado": terminated,
                "truncado": truncated,

                "erro": info_final.get(
                    "erro",
                    "",
                ),
            })

            melhoria_texto = (
                f"{melhoria_percentual:.4f}%"
                if melhoria_percentual is not None
                else "N/A"
            )

            custo_texto = (
                f"{custo_final:.2f}"
                if custo_final is not None
                and np.isfinite(custo_final)
                else "inf/None"
            )

            loss_texto = (
                f"{loss_medio:.6f}"
                if loss_medio is not None
                else "N/A"
            )

            print(
                f"Episódio {episodio:03d} | "
                f"passos={passos} | "
                f"finalizado={terminated} | "
                f"truncado={truncated} | "
                f"reward={recompensa_total:.6f} | "
                f"epsilon={epsilon:.4f} | "
                f"loss={loss_texto} | "
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
    campos = list(
        resultados[0].keys()
    )

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
    # SALVA MODELO
    # =================================================================
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_size": state_size,
            "action_size": action_size,
            "seed": SEED,
            "episodios": EPISODIOS,
            "gamma": GAMMA,
            "lr": LR,
            "epsilon_final": epsilon,
        },
        ARQUIVO_MODELO,
    )

    print("\n" + "=" * 70)
    print("TREINAMENTO FINALIZADO")
    print("=" * 70)

    print(f"CSV salvo: {ARQUIVO_SAIDA}")
    print(f"Modelo salvo: {ARQUIVO_MODELO}")

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


if __name__ == "__main__":
    executar_ddqn()
