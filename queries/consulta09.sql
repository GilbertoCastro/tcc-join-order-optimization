-- Elenco em episódios de séries populares
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 09: title_episode + title_basics(child) + title_basics(parent) + title_principals + name_basics + title_ratings
-- Objetivo: consulta com 6 aliases e múltiplos joins relacionais.
SELECT
    parent."primaryTitle" AS serie,
    nb."primaryName" AS participante,
    tp.category,
    COUNT(DISTINCT child.tconst) AS qtd_episodios,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating
FROM title_episode AS te
INNER JOIN title_basics AS child
    ON te.tconst = child.tconst
INNER JOIN title_basics AS parent
    ON te."parentTconst" = parent.tconst
INNER JOIN title_principals AS tp
    ON child.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_ratings AS tr
    ON child.tconst = tr.tconst
WHERE tr."numVotes" >= 100
  AND tp.category IN ('actor', 'actress', 'self')
GROUP BY parent."primaryTitle", nb."primaryName", tp.category
HAVING COUNT(DISTINCT child.tconst) >= 3
ORDER BY qtd_episodios DESC, media_rating DESC
LIMIT 100;
