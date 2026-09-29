-- Categorias em séries e episódios
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 24: title_episode + title_basics(parent) + title_basics(child) + title_principals + name_basics
-- Objetivo: distribuição de categorias em episódios.
SELECT
    parent."primaryTitle" AS serie,
    tp.category,
    COUNT(DISTINCT nb.nconst) AS qtd_pessoas,
    COUNT(DISTINCT child.tconst) AS qtd_episodios
FROM title_episode AS te
INNER JOIN title_basics AS child
    ON te.tconst = child.tconst
INNER JOIN title_basics AS parent
    ON te."parentTconst" = parent.tconst
INNER JOIN title_principals AS tp
    ON child.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
GROUP BY parent."primaryTitle", tp.category
HAVING COUNT(DISTINCT child.tconst) >= 5
ORDER BY qtd_episodios DESC, qtd_pessoas DESC
LIMIT 100;
