-- Títulos de TV com episódio e rating
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 07: title_basics(child) + title_episode + title_basics(parent) + title_ratings
-- Objetivo: avaliar joins com auto-relacionamento entre episódio e série.
SELECT
    parent."primaryTitle" AS serie,
    child."primaryTitle" AS episodio,
    te."seasonNumber" AS temporada,
    te."episodeNumber" AS numero_episodio,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_episode AS te
INNER JOIN title_basics AS child
    ON te.tconst = child.tconst
INNER JOIN title_basics AS parent
    ON te."parentTconst" = parent.tconst
INNER JOIN title_ratings AS tr
    ON child.tconst = tr.tconst
WHERE child."titleType" = 'tvEpisode'
  AND tr."numVotes" >= 100
ORDER BY tr."averageRating" DESC, tr."numVotes" DESC
LIMIT 100;
