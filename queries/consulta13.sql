-- Títulos recentes por profissão principal
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 13: title_principals + name_basics + title_basics + title_ratings
-- Objetivo: avaliar filtros na tabela de pessoas e títulos recentes.
SELECT
    nb."primaryProfession" AS profissao_principal,
    tp.category,
    COUNT(DISTINCT tb.tconst) AS qtd_titulos,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating
FROM title_principals AS tp
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_basics AS tb
    ON tp.tconst = tb.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."startYear" >= 2015
  AND nb."primaryProfession" IS NOT NULL
GROUP BY nb."primaryProfession", tp.category
ORDER BY qtd_titulos DESC, media_rating DESC
LIMIT 100;
