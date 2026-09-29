-- Filmes longos bem avaliados
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 02: title_basics + title_ratings
-- Objetivo: avaliar filtro por duração e avaliação.
SELECT
    tb.tconst,
    tb."primaryTitle" AS titulo,
    tb."runtimeMinutes" AS duracao_minutos,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_basics AS tb
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" = 'movie'
  AND tb."runtimeMinutes" >= 120
  AND tr."averageRating" >= 8.0
ORDER BY tb."runtimeMinutes" DESC, tr."averageRating" DESC
LIMIT 100;
