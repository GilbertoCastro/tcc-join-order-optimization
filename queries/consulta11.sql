-- Filmes adultos excluídos com equipe e avaliação
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 11: title_basics + title_crew + title_ratings
-- Objetivo: avaliar filtros booleanos e joins.
SELECT
    tb."startYear" AS ano,
    COUNT(DISTINCT tb.tconst) AS qtd_titulos,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating,
    SUM(tr."numVotes") AS votos_totais
FROM title_basics AS tb
INNER JOIN title_crew AS tc
    ON tb.tconst = tc.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" = 'movie'
  AND COALESCE(tb."isAdult", false) = false
  AND tc.directors IS NOT NULL
  AND tr."numVotes" >= 1000
GROUP BY tb."startYear"
ORDER BY ano DESC;
