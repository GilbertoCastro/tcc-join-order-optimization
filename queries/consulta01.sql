-- Top filmes por rating com votos mínimos
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 01: title_basics + title_ratings
-- Objetivo: avaliar join simples entre títulos e avaliações.
SELECT
    tb.tconst,
    tb."primaryTitle" AS titulo,
    tb."startYear" AS ano,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_basics AS tb
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" = 'movie'
  AND tr."numVotes" >= 10000
ORDER BY tr."averageRating" DESC, tr."numVotes" DESC
LIMIT 50;
