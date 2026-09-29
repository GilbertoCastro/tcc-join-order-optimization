-- Melhores episódios por participante
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 25: title_episode + child + parent + title_principals + name_basics + title_ratings
-- Objetivo: 6 aliases com rating e participante.
SELECT
    nb."primaryName" AS participante,
    parent."primaryTitle" AS serie,
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
WHERE tr."numVotes" >= 50
GROUP BY nb."primaryName", parent."primaryTitle"
HAVING COUNT(DISTINCT child.tconst) >= 3
ORDER BY media_rating DESC, qtd_episodios DESC
LIMIT 100;
