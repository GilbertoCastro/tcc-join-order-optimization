-- Atrizes em filmes bem avaliados
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 04: title_basics + title_principals + name_basics + title_ratings
-- Objetivo: avaliar join com filtro de categoria actress.
SELECT
    nb.nconst,
    nb."primaryName" AS atriz,
    COUNT(DISTINCT tb.tconst) AS qtd_filmes,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating,
    SUM(tr."numVotes") AS votos_totais
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" = 'movie'
  AND tp.category = 'actress'
  AND tr."averageRating" >= 7.0
GROUP BY nb.nconst, nb."primaryName"
HAVING COUNT(DISTINCT tb.tconst) >= 2
ORDER BY media_rating DESC, votos_totais DESC
LIMIT 50;
