-- Benchmark complexo geral com 6 aliases
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 29: title_episode + child + parent + title_principals + name_basics + title_ratings
-- Objetivo: consulta complexa para avaliar custo de múltiplos joins.
SELECT
    parent."primaryTitle" AS serie,
    child."primaryTitle" AS episodio,
    nb."primaryName" AS participante,
    tp.category,
    te."seasonNumber" AS temporada,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_episode AS te
INNER JOIN title_basics AS child
    ON te.tconst = child.tconst
INNER JOIN title_basics AS parent
    ON te."parentTconst" = parent.tconst
INNER JOIN title_principals AS tp
    ON child.tconst = tp.tconst
INNER JOIN name_basics AS nb
    ON tp.nconst = nb.nconst
INNER JOIN title_ratings AS tr
    ON child.tconst = tr.tconst
WHERE parent."titleType" IN ('tvSeries', 'tvMiniSeries')
  AND child."titleType" = 'tvEpisode'
  AND tr."averageRating" >= 8.0
  AND tr."numVotes" >= 100
ORDER BY tr."averageRating" DESC, tr."numVotes" DESC
LIMIT 200;
