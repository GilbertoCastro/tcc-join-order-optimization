-- Séries com média de ratings dos episódios
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 08: title_episode + title_basics(child) + title_basics(parent) + title_ratings
-- Objetivo: agregação por série usando joins entre episódios e ratings.
SELECT
    parent.tconst AS serie_id,
    parent."primaryTitle" AS serie,
    COUNT(DISTINCT child.tconst) AS qtd_episodios_avaliados,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating_episodios,
    SUM(tr."numVotes") AS votos_totais
FROM title_episode AS te
INNER JOIN title_basics AS child
    ON te.tconst = child.tconst
INNER JOIN title_basics AS parent
    ON te."parentTconst" = parent.tconst
INNER JOIN title_ratings AS tr
    ON child.tconst = tr.tconst
WHERE child."titleType" = 'tvEpisode'
GROUP BY parent.tconst, parent."primaryTitle"
HAVING COUNT(DISTINCT child.tconst) >= 10
ORDER BY media_rating_episodios DESC, votos_totais DESC
LIMIT 50;
