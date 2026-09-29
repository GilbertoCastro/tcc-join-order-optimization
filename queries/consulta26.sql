-- Filmes por década, categoria e rating
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 26: title_basics + title_principals + title_ratings
-- Objetivo: agrupamento por década com joins.
SELECT
    (tb."startYear" / 10) * 10 AS decada,
    tp.category,
    COUNT(DISTINCT tb.tconst) AS qtd_filmes,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" = 'movie'
  AND tb."startYear" IS NOT NULL
GROUP BY decada, tp.category
ORDER BY decada DESC, qtd_filmes DESC;
