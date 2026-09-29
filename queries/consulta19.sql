-- Títulos com maior quantidade de participantes
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 19: title_basics + title_principals + title_ratings
-- Objetivo: títulos com maior elenco/equipe registrada.
SELECT
    tb.tconst,
    tb."primaryTitle" AS titulo,
    tb."titleType" AS tipo,
    COUNT(tp.nconst) AS qtd_participantes,
    tr."averageRating" AS nota_media,
    tr."numVotes" AS total_votos
FROM title_basics AS tb
INNER JOIN title_principals AS tp
    ON tb.tconst = tp.tconst
INNER JOIN title_ratings AS tr
    ON tb.tconst = tr.tconst
WHERE tr."numVotes" >= 1000
GROUP BY tb.tconst, tb."primaryTitle", tb."titleType", tr."averageRating", tr."numVotes"
ORDER BY qtd_participantes DESC, tr."numVotes" DESC
LIMIT 100;
