import os
import re
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import psycopg2


class LearningJob(gym.Env):
    """
    Ambiente Gymnasium para estudo de ordenação de JOINs com PostgreSQL.

    Correções principais desta versão:
    1. A ordem escolhida pelo agente é materializada em uma NOVA SQL.
    2. O EXPLAIN é executado sobre a SQL reordenada, não sobre a SQL original.
    3. O histórico real das ações é preservado em action_history.
    4. As condições de JOIN são preservadas e reutilizadas na reconstrução.
    5. A consulta SQL final usada no EXPLAIN é registrada no info do episódio.
    6. O reset aceita uma consulta fixa via options, facilitando avaliação pareada.
    7. LEFT/RIGHT/FULL/CROSS JOIN são rejeitados, pois a reordenação arbitrária
       desses tipos de JOIN pode alterar a semântica da consulta.

    Observação:
    - O ambiente trabalha com INNER JOIN / JOIN relacional.
    - O PostgreSQL continua responsável pela escolha dos algoritmos físicos
      de junção, scans, índices e demais detalhes do plano.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: str = "5432",
        db_name: str = "IMDB",
        db_user: str = "postgres",
        db_password: Optional[str] = Nome,
        diretorio_consultas: Optional[str] = None,        
        
    ):
        super().__init__()

        # ================================================================
        # 1. ESQUEMA DO BANCO
        # ================================================================
        self.schema: Dict[str, List[str]] = {
            "name_basics": [
                "nconst", "primaryName", "birthYear", "deathYear",
                "primaryProfession", "knownForTitles"
            ],
            "title_basics": [
                "tconst", "titleType", "primaryTitle", "originalTitle",
                "isAdult", "startYear", "endYear", "runtimeMinutes", "genres"
            ],
            "title_crew": [
                "tconst", "directors", "writer"
            ],
            "title_episode": [
                "tconst", "parentTconst", "seasonNumber", "episodeNumber"
            ],
            "title_principals": [
                "tconst", "ordering", "nconst", "category", "job", "characters"
            ],
            "title_ratings": [
                "tconst", "averageRating", "numVotes"
            ],
        }

        self.max_tables = len(self.schema)
        self.max_actions = self.max_tables * (self.max_tables - 1) // 2

        # ================================================================
        # 2. CONEXÃO COM POSTGRESQL
        # ================================================================
        # Para não deixar senha gravada no arquivo:
        #   PowerShell:
        #   $env:IMDB_DB_PASSWORD="sua_senha"
        #
        # Ou passe db_password diretamente ao criar LearningJob(...).
        if db_password is None:
            db_password = os.getenv("IMDB_DB_PASSWORD")

        if not db_password:
            raise ValueError(
                "Senha do PostgreSQL não informada. Defina a variável de ambiente "
                "IMDB_DB_PASSWORD ou passe db_password ao criar LearningJob."
            )

        self.conexao = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            client_encoding="utf8",
        )
        self.cursor = self.conexao.cursor()

        # Importante para que a ordem explícita de JOIN tenha influência.
        self.cursor.execute("SET join_collapse_limit = 1;")
        self.cursor.execute("SET from_collapse_limit = 1;")

        # ================================================================
        # 3. CONSULTAS
        # ================================================================
        if diretorio_consultas is None:
            diretorio_consultas = ("caminho_dos_arquivos_de_query"
                
            )

        self.diretorio_consultas = diretorio_consultas
        self.consultas_sql = self._load_queries(self.diretorio_consultas)

        print(
            f"Total de consultas benchmark JOIN carregadas: "
            f"{len(self.consultas_sql)}"
        )

        if not self.consultas_sql:
            raise FileNotFoundError(
                "Nenhuma consulta SQL válida foi carregada. Confira o diretório "
                "e os arquivos consulta01.sql ... consulta29.sql."
            )

        # ================================================================
        # 4. ESPAÇOS GYMNASIUM
        # ================================================================
        obs_size = self.max_tables + self.max_actions

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.max_actions)

        # ================================================================
        # 5. VARIÁVEIS DE EPISÓDIO
        # ================================================================
        self.consulta_atual: str = ""
        self.query_prefix: str = ""
        self.query_suffix: str = ""

        # Alias -> nome simples da tabela (para validação contra schema)
        self.alias_to_table: Dict[str, str] = {}

        # Alias -> referência SQL original da tabela, preservando schema se houver
        self.alias_to_table_ref: Dict[str, str] = {}

        # Mantido por compatibilidade
        self.table_to_alias: Dict[str, str] = {}

        self.aliases: List[str] = []

        # Arestas relacionais disponíveis
        self.join_edges: Set[Tuple[str, str]] = set()

        # (alias_a, alias_b) -> lista de predicados SQL originais
        self.join_conditions: Dict[Tuple[str, str], List[str]] = {}

        self.action_pairs: List[Tuple[str, str]] = []
        self.components: List[Set[str]] = []

        # Para cada componente, mantém a expressão SQL correspondente.
        # A chave é frozenset(component).
        self.component_sql: Dict[frozenset, str] = {}

        self.used_actions: Set[int] = set()

        # Histórico real, na ordem em que as ações foram executadas.
        self.action_history: List[int] = []

        self.joined_aliases: Set[str] = set()
        self.steps_taken: int = 0

        # SQL reordenada construída ao fim do episódio
        self.consulta_reordenada_final: Optional[str] = None

    # ================================================================
    # CARREGAMENTO E LIMPEZA
    # ================================================================
    def _load_queries(self, directory: str) -> List[str]:
        queries: List[str] = []

        for i in range(1, 30):
            filename = f"consulta{i:02d}.sql"
            path = os.path.join(directory, filename)

            if not os.path.exists(path):
                print(f"Aviso: arquivo não encontrado: {path}")
                continue

            with open(path, "r", encoding="utf-8") as file:
                raw_sql = file.read()

            sql = self._clean_sql(raw_sql)

            if not sql:
                continue

            if not re.search(r"\bJOIN\b", sql, re.IGNORECASE):
                print(f"Ignorada (sem JOIN): {filename}")
                continue

            if self._has_unsupported_join_type(sql):
                print(
                    f"Ignorada ({filename}): contém LEFT/RIGHT/FULL/CROSS JOIN. "
                    "Esta versão reordena apenas INNER JOIN."
                )
                continue

            (
                alias_to_table,
                alias_to_table_ref,
                join_edges,
                join_conditions,
            ) = self.parse_query_detailed(sql)

            alias_to_table = {
                alias: table
                for alias, table in alias_to_table.items()
                if table in self.schema
            }

            alias_to_table_ref = {
                alias: table_ref
                for alias, table_ref in alias_to_table_ref.items()
                if alias in alias_to_table
            }

            join_edges = {
                edge
                for edge in join_edges
                if edge[0] in alias_to_table and edge[1] in alias_to_table
            }

            join_conditions = {
                edge: conditions
                for edge, conditions in join_conditions.items()
                if edge in join_edges
            }

            if not self._is_connected_join_graph(alias_to_table, join_edges):
                print(f"Ignorada ({filename}): grafo JOIN desconectado.")
                continue

            # Toda aresta usada pelo ambiente precisa possuir condição SQL.
            edges_sem_condicao = [
                edge for edge in join_edges
                if not join_conditions.get(edge)
            ]
            if edges_sem_condicao:
                print(
                    f"Ignorada ({filename}): existem arestas sem condição SQL "
                    f"recuperável: {edges_sem_condicao}"
                )
                continue

            queries.append(sql)

        return queries

    @staticmethod
    def _clean_sql(sql: str) -> str:
        """Remove comentários e normaliza espaços."""
        sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
        sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
        sql = sql.replace("\n", " ").replace("\t", " ")
        sql = re.sub(r"\s+", " ", sql).strip()
        return sql.rstrip(";")

    @staticmethod
    def _has_unsupported_join_type(query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:LEFT|RIGHT|FULL|CROSS)\s+(?:OUTER\s+)?JOIN\b",
                query,
                flags=re.IGNORECASE,
            )
        )

    # ================================================================
    # PARSER
    # ================================================================
    @staticmethod
    def _normalize_edge(a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))

    def _extract_table_references(
        self,
        query: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Retorna:
            alias_to_table:
                {"tb": "title_basics"}

            alias_to_table_ref:
                {"tb": "public.title_basics"}
        """
        alias_to_table: Dict[str, str] = {}
        alias_to_table_ref: Dict[str, str] = {}

        table_pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+"
            r"([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)?)"
            r"(?:\s+(?:AS\s+)?([a-zA-Z_][\w]*))?",
            flags=re.IGNORECASE,
        )

        reserved = {
            "inner", "left", "right", "full", "cross", "join", "on", "where",
            "group", "order", "limit", "having", "union", "offset", "fetch",
        }

        for table_ref, alias in table_pattern.findall(query):
            clean_table = table_ref.split(".")[-1]
            chosen_alias = (
                alias
                if alias and alias.lower() not in reserved
                else clean_table
            )

            alias_to_table[chosen_alias] = clean_table
            alias_to_table_ref[chosen_alias] = table_ref

        return alias_to_table, alias_to_table_ref

    @staticmethod
    def _extract_relational_predicates(text: str) -> List[Tuple[str, str, str]]:
        """
        Extrai predicados simples de igualdade entre colunas de dois aliases.

        Exemplos aceitos:
            tb.tconst = tr.tconst
            tr.tconst=tp.tconst

        Retorna:
            [(alias_esq, alias_dir, "tb.tconst = tr.tconst"), ...]
        """
        pattern = re.compile(
            r"([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)"
            r"\s*=\s*"
            r"([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)",
            flags=re.IGNORECASE,
        )

        predicates: List[Tuple[str, str, str]] = []

        for match in pattern.finditer(text):
            left_alias, left_col, right_alias, right_col = match.groups()

            if left_alias == right_alias:
                continue

            predicate = (
                f"{left_alias}.{left_col} = "
                f"{right_alias}.{right_col}"
            )

            predicates.append(
                (left_alias, right_alias, predicate)
            )

        return predicates

    def parse_query_detailed(
        self,
        query: str,
    ) -> Tuple[
        Dict[str, str],
        Dict[str, str],
        Set[Tuple[str, str]],
        Dict[Tuple[str, str], List[str]],
    ]:
        """
        Extrai tabelas, aliases, arestas e predicados relacionais.

        IMPORTANTE:
        A reconstrução usa os predicados de igualdade entre colunas para montar
        novamente os INNER JOINs.

        Retorna:
            alias_to_table
            alias_to_table_ref
            join_edges
            join_conditions
        """
        alias_to_table, alias_to_table_ref = self._extract_table_references(query)

        join_edges: Set[Tuple[str, str]] = set()
        join_conditions: Dict[Tuple[str, str], List[str]] = {}

        for left_alias, right_alias, predicate in self._extract_relational_predicates(query):
            if left_alias not in alias_to_table or right_alias not in alias_to_table:
                continue

            edge = self._normalize_edge(left_alias, right_alias)
            join_edges.add(edge)
            join_conditions.setdefault(edge, [])

            if predicate not in join_conditions[edge]:
                join_conditions[edge].append(predicate)

        return (
            alias_to_table,
            alias_to_table_ref,
            join_edges,
            join_conditions,
        )

    def parse_query(
        self,
        query: str
    ) -> Tuple[Dict[str, str], Set[Tuple[str, str]]]:
        """
        Mantém compatibilidade com o código anterior.
        """
        alias_to_table, _, join_edges, _ = self.parse_query_detailed(query)
        return alias_to_table, join_edges

    def get_actions(self, query: str, ids=None, masks=None, schema=None):
        """Compatibilidade com a função antiga."""
        alias_to_table, join_edges = self.parse_query(query)
        actions = list(alias_to_table.keys())
        joins = {"_".join(edge): True for edge in join_edges}
        return actions, joins

    # ================================================================
    # DIVISÃO DA SQL EM SELECT / FROM / SUFIXO
    # ================================================================
    @staticmethod
    def _split_query(query: str) -> Tuple[str, str]:
        """
        Divide a consulta em:
            prefixo: tudo antes do FROM
            sufixo: WHERE/GROUP BY/HAVING/ORDER BY/LIMIT/OFFSET/FETCH/UNION...

        O FROM original será substituído pela expressão reordenada.
        """
        from_match = re.search(r"\bFROM\b", query, flags=re.IGNORECASE)

        if not from_match:
            raise ValueError("Consulta sem cláusula FROM.")

        prefix = query[:from_match.start()].strip()

        remainder = query[from_match.end():].strip()

        suffix_match = re.search(
            r"\b(WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET|FETCH|UNION)\b",
            remainder,
            flags=re.IGNORECASE,
        )

        if suffix_match:
            suffix = remainder[suffix_match.start():].strip()
        else:
            suffix = ""

        return prefix, suffix

    # ================================================================
    # GRAFO
    # ================================================================
    def _is_connected_join_graph(
        self,
        alias_to_table: Dict[str, str],
        join_edges: Set[Tuple[str, str]],
    ) -> bool:
        aliases = set(alias_to_table.keys())

        if len(aliases) <= 1 or not join_edges:
            return False

        graph = {alias: set() for alias in aliases}

        for a, b in join_edges:
            if a in graph and b in graph:
                graph[a].add(b)
                graph[b].add(a)

        start = next(iter(aliases))
        visited: Set[str] = set()
        stack = [start]

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            stack.extend(graph[current] - visited)

        return visited == aliases

    # ================================================================
    # MÉTRICAS POSTGRESQL
    # ================================================================
    def estimate_plan(self, query: str) -> Dict[str, Optional[float]]:
        """
        Executa EXPLAIN (FORMAT JSON), sem executar a consulta real.
        """
        try:
            self.cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
            result = self.cursor.fetchone()[0]
            plan = result[0]["Plan"]

            return {
                "startup_cost": float(plan.get("Startup Cost", 0.0)),
                "total_cost": float(plan.get("Total Cost", 0.0)),
                "plan_rows": float(plan.get("Plan Rows", 0.0)),
                "plan_width": float(plan.get("Plan Width", 0.0)),
            }

        except Exception as exc:
            self.conexao.rollback()

            # O rollback também desfaz SETs locais da transação anterior em alguns
            # contextos; reaplicamos os parâmetros de interesse.
            try:
                self.cursor.execute("SET join_collapse_limit = 1;")
                self.cursor.execute("SET from_collapse_limit = 1;")
            except Exception:
                pass

            return {
                "startup_cost": None,
                "total_cost": None,
                "plan_rows": None,
                "plan_width": None,
                "error": str(exc),
            }

    def estimate_cost(self, query: str) -> float:
        plan = self.estimate_plan(query)

        if plan.get("total_cost") is None:
            return float("inf")

        return float(plan["total_cost"])

    def estimate_cardinality(self, query: str) -> float:
        plan = self.estimate_plan(query)

        if plan.get("plan_rows") is None:
            return float("inf")

        return float(plan["plan_rows"])

    # ================================================================
    # PREPARAÇÃO DO EPISÓDIO
    # ================================================================
    def _prepare_query_for_episode(
        self,
        query: str,
    ) -> Tuple[
        Dict[str, str],
        Dict[str, str],
        Set[Tuple[str, str]],
        Dict[Tuple[str, str], List[str]],
    ]:
        (
            alias_to_table,
            alias_to_table_ref,
            join_edges,
            join_conditions,
        ) = self.parse_query_detailed(query)

        alias_to_table = {
            alias: table
            for alias, table in alias_to_table.items()
            if table in self.schema
        }

        alias_to_table_ref = {
            alias: table_ref
            for alias, table_ref in alias_to_table_ref.items()
            if alias in alias_to_table
        }

        join_edges = {
            edge
            for edge in join_edges
            if edge[0] in alias_to_table and edge[1] in alias_to_table
        }

        join_conditions = {
            edge: conditions
            for edge, conditions in join_conditions.items()
            if edge in join_edges
        }

        return (
            alias_to_table,
            alias_to_table_ref,
            join_edges,
            join_conditions,
        )

    def _choose_query(self, options: Optional[dict]) -> str:
        """
        Permite:
            env.reset()
                -> consulta aleatória

            env.reset(options={"query": sql})
                -> consulta específica

            env.reset(options={"query_index": 0})
                -> consulta específica pelo índice da lista carregada

        Isso será útil para avaliação pareada.
        """
        if options:
            if "query" in options and options["query"]:
                return self._clean_sql(str(options["query"]))

            if "query_index" in options:
                idx = int(options["query_index"])

                if idx < 0 or idx >= len(self.consultas_sql):
                    raise IndexError(
                        f"query_index fora do intervalo: {idx}"
                    )

                return self.consultas_sql[idx]

        idx = int(self.np_random.integers(0, len(self.consultas_sql)))
        return self.consultas_sql[idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        tentativas = 0

        while True:
            self.consulta_atual = self._choose_query(options)

            (
                self.alias_to_table,
                self.alias_to_table_ref,
                self.join_edges,
                self.join_conditions,
            ) = self._prepare_query_for_episode(self.consulta_atual)

            if self._is_connected_join_graph(
                self.alias_to_table,
                self.join_edges,
            ):
                break

            # Se a consulta foi fixada pelo usuário, não há sentido em sortear outra.
            if options and (
                options.get("query") is not None
                or options.get("query_index") is not None
            ):
                raise ValueError(
                    "A consulta fornecida não possui grafo de JOIN conectado "
                    "e compatível com o ambiente."
                )

            tentativas += 1

            if tentativas > 50:
                raise RuntimeError(
                    "Não foi possível encontrar consulta com grafo JOIN conectado "
                    "após 50 tentativas."
                )

        self.query_prefix, self.query_suffix = self._split_query(self.consulta_atual)

        self.aliases = list(self.alias_to_table.keys())[:self.max_tables]

        self.table_to_alias = {
            table: alias
            for alias, table in self.alias_to_table.items()
        }

        self.action_pairs = list(combinations(self.aliases, 2))[:self.max_actions]

        self.components = [{alias} for alias in self.aliases]

        # Inicializa cada componente com a referência SQL da tabela e seu alias.
        self.component_sql = {}

        for alias in self.aliases:
            table_ref = self.alias_to_table_ref[alias]

            if alias == table_ref.split(".")[-1]:
                # Mesmo quando alias = nome da tabela, manter alias explícito ajuda
                # a preservar referências como alias.coluna.
                sql_fragment = f"{table_ref} AS {alias}"
            else:
                sql_fragment = f"{table_ref} AS {alias}"

            self.component_sql[frozenset({alias})] = sql_fragment

        self.used_actions = set()
        self.action_history = []
        self.joined_aliases = set()
        self.steps_taken = 0
        self.consulta_reordenada_final = None

        observation = self._build_observation()

        custo_original = self.estimate_cost(self.consulta_atual)
        cardinalidade_original = self.estimate_cardinality(self.consulta_atual)

        info = {
            "consulta_selecionada": self.consulta_atual,
            "aliases": self.alias_to_table,
            "joins_validos": sorted(list(self.join_edges)),
            "condicoes_join": self.join_conditions,
            "action_pairs": self.action_pairs,
            "custo_postgresql": custo_original,
            "cardinalidade_postgresql": cardinalidade_original,
        }

        return observation, info

    # ================================================================
    # RECONSTRUÇÃO SQL
    # ================================================================
    def _conditions_between_components(
        self,
        left_component: Set[str],
        right_component: Set[str],
    ) -> List[str]:
        """
        Retorna todos os predicados originais que cruzam os dois componentes.
        Isso preserva múltiplas condições entre as relações.
        """
        conditions: List[str] = []

        for edge, predicates in self.join_conditions.items():
            a, b = edge

            crosses = (
                (a in left_component and b in right_component)
                or
                (b in left_component and a in right_component)
            )

            if not crosses:
                continue

            for predicate in predicates:
                if predicate not in conditions:
                    conditions.append(predicate)

        return conditions

    def _merge_components_sql(
        self,
        left_component: Set[str],
        right_component: Set[str],
    ) -> Tuple[Set[str], str]:
        """
        Une dois componentes gerando um novo fragmento SQL parentetizado.
        """
        left_key = frozenset(left_component)
        right_key = frozenset(right_component)

        left_sql = self.component_sql[left_key]
        right_sql = self.component_sql[right_key]

        conditions = self._conditions_between_components(
            left_component,
            right_component,
        )

        if not conditions:
            raise RuntimeError(
                "Tentativa de unir componentes sem condição relacional. "
                f"Esquerda={left_component}, Direita={right_component}"
            )

        on_clause = " AND ".join(f"({c})" for c in conditions)

        merged_sql = (
            f"({left_sql} INNER JOIN {right_sql} ON {on_clause})"
        )

        merged_component = set(left_component) | set(right_component)

        # Remove fragments antigos e registra o novo.
        del self.component_sql[left_key]
        del self.component_sql[right_key]

        self.component_sql[frozenset(merged_component)] = merged_sql

        return merged_component, merged_sql

    def build_reordered_query(self) -> str:
        """
        Retorna a SQL completa correspondente à árvore de JOIN construída
        pelas ações executadas até o momento.
        """
        if len(self.components) != 1:
            raise RuntimeError(
                "A consulta só pode ser reconstruída após conectar "
                "todos os componentes."
            )

        component = frozenset(self.components[0])

        if component not in self.component_sql:
            raise RuntimeError(
                "Fragmento SQL final não encontrado para o componente terminal."
            )

        from_expression = self.component_sql[component]

        if self.query_suffix:
            return (
                f"{self.query_prefix} FROM {from_expression} "
                f"{self.query_suffix}"
            )

        return f"{self.query_prefix} FROM {from_expression}"

    # ================================================================
    # AÇÕES
    # ================================================================
    def valid_action_indices(self) -> List[int]:
        """
        Retorna as ações válidas no estado atual.
        Útil para Random, Greedy, DQN, DDQN e para mascaramento no replay.
        """
        valid: List[int] = []

        for idx, pair in enumerate(self.action_pairs):
            if idx in self.used_actions:
                continue

            edge = self._normalize_edge(*pair)

            if edge not in self.join_edges:
                continue

            left_alias, right_alias = pair

            try:
                left_component_index = self._find_component_index(left_alias)
                right_component_index = self._find_component_index(right_alias)
            except ValueError:
                continue

            if left_component_index == right_component_index:
                continue

            valid.append(idx)

        return valid

    def step(self, action: int):
        self.steps_taken += 1

        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        if action < 0 or action >= len(self.action_pairs):
            reward = -100.0
            info["erro"] = "Ação fora dos pares disponíveis."
            return (
                self._build_observation(),
                reward,
                terminated,
                truncated,
                info,
            )

        if action in self.used_actions:
            reward = -20.0
            info["erro"] = "Ação repetida."
            return (
                self._build_observation(),
                reward,
                terminated,
                truncated,
                info,
            )

        left_alias, right_alias = self.action_pairs[action]
        edge = self._normalize_edge(left_alias, right_alias)

        if edge not in self.join_edges:
            reward = -100.0
            info["erro"] = (
                "Junção cruzada bloqueada. "
                "Não existe condição relacional para esse par."
            )
            info["par"] = edge

            return (
                self._build_observation(),
                reward,
                terminated,
                truncated,
                info,
            )

        left_component_index = self._find_component_index(left_alias)
        right_component_index = self._find_component_index(right_alias)

        if left_component_index == right_component_index:
            reward = -10.0
            info["erro"] = (
                "Os aliases já pertencem à mesma subárvore."
            )

            return (
                self._build_observation(),
                reward,
                terminated,
                truncated,
                info,
            )

        left_component = set(self.components[left_component_index])
        right_component = set(self.components[right_component_index])

        # Constrói o fragmento SQL ANTES de modificar a lista de componentes.
        try:
            merged_component, _ = self._merge_components_sql(
                left_component,
                right_component,
            )
        except Exception as exc:
            reward = -1000.0
            truncated = True
            info["erro"] = f"Erro ao reconstruir JOIN: {exc}"

            return (
                self._build_observation(),
                reward,
                terminated,
                truncated,
                info,
            )

        # Remove sempre do maior índice primeiro para evitar deslocamento.
        for idx in sorted(
            [left_component_index, right_component_index],
            reverse=True,
        ):
            del self.components[idx]

        self.components.append(merged_component)

        self.used_actions.add(action)
        self.action_history.append(action)
        self.joined_aliases.update([left_alias, right_alias])

        # Pequeno ganho por ação estruturalmente válida.
        reward = 1.0

        if len(self.components) == 1:
            terminated = True

            try:
                reordered_query = self.build_reordered_query()
                self.consulta_reordenada_final = reordered_query

                plan = self.estimate_plan(reordered_query)

                if plan.get("total_cost") is None:
                    cost = float("inf")
                    cardinality = float("inf")
                    plan_error = plan.get("error", "Erro desconhecido no EXPLAIN.")
                else:
                    cost = float(plan["total_cost"])
                    cardinality = float(plan["plan_rows"])
                    plan_error = ""

                if np.isfinite(cost):
                    reward += -np.log1p(cost)
                else:
                    reward += -1000.0

                info["episodio_finalizado"] = True
                info["consulta_sql_reordenada"] = reordered_query
                info["custo_estimado_final"] = cost
                info["cardinalidade_estimada_final"] = cardinality
                info["ordem_logica_joins"] = [
                    self.action_pairs[i]
                    for i in self.action_history
                ]

                if plan_error:
                    info["erro"] = plan_error

            except Exception as exc:
                reward += -1000.0
                info["episodio_finalizado"] = False
                info["erro"] = f"Erro na avaliação final: {exc}"
                info["consulta_sql_reordenada"] = (
                    self.consulta_reordenada_final or ""
                )
                info["custo_estimado_final"] = float("inf")
                info["cardinalidade_estimada_final"] = float("inf")
                info["ordem_logica_joins"] = [
                    self.action_pairs[i]
                    for i in self.action_history
                ]

        if self.steps_taken >= self.max_actions and not terminated:
            truncated = True
            reward -= 50.0
            info["truncated"] = "Limite máximo de ações atingido."

        return (
            self._build_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ================================================================
    # OBSERVAÇÃO / UTILIDADES
    # ================================================================
    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(
            self.observation_space.shape,
            dtype=np.float32,
        )

        # Parte 1: aliases já envolvidos em alguma junção.
        for i, alias in enumerate(self.aliases[:self.max_tables]):
            if alias in self.joined_aliases:
                obs[i] = 1.0

        # Parte 2: ações já executadas.
        offset = self.max_tables

        for action_index in self.used_actions:
            if offset + action_index < len(obs):
                obs[offset + action_index] = 1.0

        return obs

    def _find_component_index(self, alias: str) -> int:
        for index, component in enumerate(self.components):
            if alias in component:
                return index

        raise ValueError(
            f"Alias não encontrado nos componentes: {alias}"
        )

    def render(self):
        print("Consulta original:")
        print(self.consulta_atual)

        print("\nAliases:")
        print(self.alias_to_table)

        print("\nJoins válidos:")
        print(self.join_edges)

        print("\nCondições de JOIN:")
        print(self.join_conditions)

        print("\nComponentes:")
        print(self.components)

        print("\nHistórico de ações:")
        print([
            self.action_pairs[i]
            for i in self.action_history
        ])

        if self.consulta_reordenada_final:
            print("\nConsulta reordenada final:")
            print(self.consulta_reordenada_final)

    def close(self):
        if getattr(self, "cursor", None):
            self.cursor.close()

        if getattr(self, "conexao", None):
            self.conexao.close()
            print("Conexão com o banco encerrada pelo ambiente.")


# =====================================================================
# TESTE LOCAL
# ATENÇÃO:
# Este query_index=2 é usado SOMENTE quando este arquivo é executado
# diretamente para teste. Não interfere em DQN, DDQN ou baselines.

# =====================================================================
if __name__ == "__main__":
    print("Iniciando teste do LearningJob corrigido...")

    env = LearningJob()
    
    # Apenas teste manual:
    #state, info = env.reset(seed=42)
    state, info = env.reset(options={"query_index": 2})

    print("\n--- RESET ---")
    print("Consulta original:")
    print(info["consulta_selecionada"])
    print("\nAliases:", info["aliases"])
    print("Joins válidos:", info["joins_validos"])
    print("Condições:", info["condicoes_join"])
    print("Pares de ação:", info["action_pairs"])
    print("Custo PostgreSQL original:", info["custo_postgresql"])

    print("\n--- EXECUTANDO AÇÕES VÁLIDAS ---")

    terminated = False
    truncated = False

    while not terminated and not truncated:
        valid_actions = env.valid_action_indices()

        if not valid_actions:
            print("Nenhuma ação válida restante.")
            break

        # No teste local, escolhe a primeira ação válida.
        action = valid_actions[0]

        state, reward, terminated, truncated, step_info = env.step(action)

        print(
            f"Ação {action} {env.action_pairs[action]} | "
            f"reward={reward:.6f}"
        )

        if step_info:
            print("Info:", step_info)

    if terminated:
        print("\n--- RESULTADO FINAL ---")
        print("Ordem real executada:")
        print([
            env.action_pairs[i]
            for i in env.action_history
        ])

        print("\nConsulta ORIGINAL:")
        print(env.consulta_atual)

        print("\nConsulta REORDENADA:")
        print(env.consulta_reordenada_final)

        custo_original = info["custo_postgresql"]
        custo_final = step_info.get("custo_estimado_final")

        print("\nCusto original:", custo_original)
        print("Custo reordenado:", custo_final)

        if (
            custo_final is not None
            and np.isfinite(custo_final)
            and np.isfinite(custo_original)
        ):
            diferenca = custo_final - custo_original
            percentual = (
                (custo_original - custo_final)
                / custo_original
                * 100.0
                if custo_original != 0
                else 0.0
            )

            print("Diferença absoluta:", diferenca)
            print(f"Melhoria relativa: {percentual:.6f}%")

    env.close()
