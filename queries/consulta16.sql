-- Animações com equipe principal e avaliação
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 16: title_basics + title_principals + name_basics + title_ratings
-- Objetivo: consulta com gênero Animation.
SELECT
    nb."primaryName" AS profissional,
    tp.category,
    COUNT(DISTINCT tb.tconst) AS qtd_animacoes,
    ROUND(AVG(tr."averageRating")::numeric, 2) AS media_rating
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb.genres ILIKE '%Animation%'
  AND tr."numVotes" >= 200
GROUP BY nb."primaryName", tp.category
HAVING COUNT(DISTINCT tb.tconst) >= 2
ORDER BY qtd_animacoes DESC, media_rating DESC
LIMIT 100;
