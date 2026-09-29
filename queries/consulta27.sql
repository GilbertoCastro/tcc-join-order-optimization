-- Títulos por faixa de duração e categoria
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 27: title_basics + title_principals + title_ratings
-- Objetivo: CASE + joins.
SELECT
    CASE
        WHEN tb."runtimeMinutes" < 60 THEN 'curto'
        WHEN tb."runtimeMinutes" BETWEEN 60 AND 119 THEN 'medio'
        WHEN tb."runtimeMinutes" >= 120 THEN 'longo'
        ELSE 'sem_duracao'
    END AS faixa_duracao,
    tp.category,
    COUNT(DISTINCT tb.tconst) AS qtd_titulos,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" IN ('movie', 'tvMovie')
GROUP BY faixa_duracao, tp.category
ORDER BY qtd_titulos DESC;
