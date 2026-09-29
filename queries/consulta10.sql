-- Títulos com equipe e rating
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 10: title_basics + title_crew + title_ratings
-- Objetivo: avaliar join com title_crew, sem quebrar lista de directors/writer.
SELECT
    tb.tconst,
    tb."primaryTitle" AS titulo,
    tb."titleType" AS tipo,
    tc.directors,
    tc.writer,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_basics AS tb
INNER JOIN title_crew AS tc
    ON tb.tconst = tc.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tr."numVotes" >= 5000
  AND tb."startYear" >= 2000
ORDER BY tr."averageRating" DESC
LIMIT 100;
