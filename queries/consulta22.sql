-- Títulos com crew e principals
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 22: title_basics + title_crew + title_principals + name_basics
-- Objetivo: testar joins entre título, crew e participantes.
SELECT
    tb."primaryTitle" AS titulo,
    tb."startYear" AS ano,
    nb."primaryName" AS participante,
    tp.category,
    tc.directors,
    tc.writer
FROM title_basics AS tb
INNER JOIN title_crew AS tc
    ON tb.tconst = tc.tconst
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
WHERE tb."titleType" = 'movie'
  AND tb."startYear" >= 2010
  AND tp.category IN ('actor', 'actress', 'director')
LIMIT 200;
