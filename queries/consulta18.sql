-- Participantes conhecidos em filmes recentes
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 18: title_basics + title_principals + name_basics + title_ratings
-- Objetivo: usa knownForTitles como filtro textual, mantendo joins por igualdade.
SELECT
    nb."primaryName" AS nome,
    tb."primaryTitle" AS titulo,
    tb."startYear" AS ano,
    tp.category,
    tr."averageRating" AS nota_media
FROM title_principals AS tp
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_basics AS tb
    ON tp.tconst = tb.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb."startYear" >= 2020
  AND nb."knownForTitles" IS NOT NULL
  AND tr."averageRating" >= 7.5
ORDER BY tr."averageRating" DESC, tb."startYear" DESC
LIMIT 100;
