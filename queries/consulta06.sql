-- Comédia por elenco e rating
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 06: title_basics + title_principals + name_basics + title_ratings
-- Objetivo: avaliar joins com filtro de gênero Comedy e ano.
SELECT
    tb."startYear" AS ano,
    nb."primaryName" AS nome,
    tp.category,
    COUNT(DISTINCT tb.tconst) AS qtd_titulos,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb.genres ILIKE '%Comedy%'
  AND tb."startYear" BETWEEN 2010 AND 2024
GROUP BY tb."startYear", nb."primaryName", tp.category
HAVING COUNT(DISTINCT tb.tconst) >= 2
ORDER BY ano DESC, qtd_titulos DESC
LIMIT 100;
