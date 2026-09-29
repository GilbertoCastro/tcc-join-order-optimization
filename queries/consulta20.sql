-- Episódios de série por temporada e rating
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 20: title_episode + title_basics(child) + title_basics(parent) + title_ratings
-- Objetivo: agregação por temporada com auto-relacionamento.
SELECT
    parent."primaryTitle" AS serie,
    te."seasonNumber" AS temporada,
    COUNT(DISTINCT child.tconst) AS qtd_episodios,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating,
    SUM(tr."numVotes") AS votos_totais
FROM title_episode AS te
INNER JOIN title_basics AS child
    ON te.tconst = child.tconst
INNER JOIN title_basics AS parent
    ON te."parentTconst" = parent.tconst
INNER JOIN title_ratings AS tr
    ON child.tconst = tr.tconst
WHERE te."seasonNumber" IS NOT NULL
GROUP BY parent."primaryTitle", te."seasonNumber"
HAVING COUNT(DISTINCT child.tconst) >= 3
ORDER BY media_rating DESC, votos_totais DESC
LIMIT 100;
