-- Ranking de pessoas por votos em títulos
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 12: title_principals + name_basics + title_basics + title_ratings
-- Objetivo: agregação ponderada por votos.
SELECT
    nb.nconst,
    nb."primaryName" AS nome,
    tp.category,
    COUNT(DISTINCT tb.tconst) AS qtd_titulos,
    SUM(tr."numVotes") AS votos_totais,
    ROUND((SUM(tr."averageRating" * tr."numVotes") / NULLIF(SUM(tr."numVotes"), 0))::numeric, 2) AS media_ponderada
FROM title_principals AS tp
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_basics AS tb
    ON tp.tconst = tb.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" = 'movie'
  AND tp.category IN ('actor', 'actress')
GROUP BY nb.nconst, nb."primaryName", tp.category
HAVING COUNT(DISTINCT tb.tconst) >= 5
ORDER BY media_ponderada DESC, votos_totais DESC
LIMIT 100;
