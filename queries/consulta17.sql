-- Média por tipo de título e categoria de participante
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 17: title_basics + title_principals + title_ratings
-- Objetivo: avaliação agregada com três tabelas.
SELECT
    tb."titleType" AS tipo_titulo,
    tp.category,
    COUNT(DISTINCT tb.tconst) AS qtd_titulos,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating,
    SUM(tr."numVotes") AS votos_totais
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tr."numVotes" >= 100
GROUP BY tb."titleType", tp.category
ORDER BY qtd_titulos DESC, votos_totais DESC;
