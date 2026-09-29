-- Episódios com atores e dados da série
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 14: title_episode + title_basics(child) + title_basics(parent) + title_principals + name_basics
-- Objetivo: 5 aliases sem rating, focando árvore de joins.
SELECT
    parent."primaryTitle" AS serie,
    child."primaryTitle" AS episodio,
    nb."primaryName" AS ator_ou_atriz,
    tp.category,
    te."seasonNumber" AS temporada,
    te."episodeNumber" AS episodio_numero
FROM title_episode AS te
INNER JOIN title_basics AS child
    ON te.tconst = child.tconst
INNER JOIN title_basics AS parent
    ON te."parentTconst" = parent.tconst
INNER JOIN title_principals AS tp
    ON child.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
WHERE tp.category IN ('actor', 'actress')
  AND te."seasonNumber" IS NOT NULL
ORDER BY parent."primaryTitle", te."seasonNumber", te."episodeNumber"
LIMIT 200;
