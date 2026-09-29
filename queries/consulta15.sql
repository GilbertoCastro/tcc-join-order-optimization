-- Filmes de ação com elenco e votos
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 15: title_basics + title_principals + name_basics + title_ratings
-- Objetivo: filtro por gênero Action em string.
SELECT
    tb."primaryTitle" AS titulo,
    tb."startYear" AS ano,
    nb."primaryName" AS nome,
    tp.category,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tb.genres ILIKE '%Action%'
  AND tb."titleType" = 'movie'
  AND tr."numVotes" >= 5000
ORDER BY tr."numVotes" DESC
LIMIT 150;
