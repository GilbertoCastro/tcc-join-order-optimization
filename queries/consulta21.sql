-- Obras com criança/pessoa nascida antes de 1980
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 21: title_basics + title_principals + name_basics + title_ratings
-- Objetivo: filtro temporal em birthYear.
SELECT
    nb."primaryName" AS nome,
    nb."birthYear" AS nascimento,
    COUNT(DISTINCT tb.tconst) AS qtd_titulos,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating
FROM name_basics AS nb
INNER JOIN title_principals AS tp
    ON nb.nconst = tp.nconst
INNER JOIN title_basics AS tb
    ON tp.tconst = tb.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE nb."birthYear" IS NOT NULL
  AND nb."birthYear" < 1980
  AND tb."titleType" = 'movie'
GROUP BY nb."primaryName", nb."birthYear"
HAVING COUNT(DISTINCT tb.tconst) >= 5
ORDER BY qtd_titulos DESC, media_rating DESC
LIMIT 100;
