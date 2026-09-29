-- Títulos com crew, principals e rating
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 23: title_basics + title_crew + title_principals + name_basics + title_ratings
-- Objetivo: 5 aliases e filtros em rating.
SELECT
    tb."primaryTitle" AS titulo,
    tb."startYear" AS ano,
    nb."primaryName" AS participante,
    tp.category,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_basics AS tb
INNER JOIN title_crew AS tc
    ON tb.tconst = tc.tconst
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."titleType" = 'movie'
  AND tr."averageRating" >= 8.0
  AND tr."numVotes" >= 5000
ORDER BY tr."averageRating" DESC, tr."numVotes" DESC
LIMIT 150;
