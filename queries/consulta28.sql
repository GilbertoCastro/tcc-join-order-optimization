-- Títulos mais votados por pessoa
-- Adaptada para PostgreSQL e para o modelo IMDb oficial: name_basics, title_basics, title_crew, title_episode, title_principals, title_ratings.
-- Consulta 28: title_principals + name_basics + title_basics + title_ratings
-- Objetivo: window function com joins.
WITH pessoa_titulo AS (
    SELECT
        nb.nconst,
        nb."primaryName" AS nome,
        tb.tconst,
        tb."primaryTitle" AS titulo,
        tr."numVotes" AS votos,
        tr."averageRating" AS nota,
        ROW_NUMBER() OVER (PARTITION BY nb.nconst ORDER BY tr."numVotes" DESC) AS rn
    FROM title_principals AS tp
    INNER JOIN name_basics AS nb
        ON tp.nconst = nb.nconst
    INNER JOIN title_basics AS tb
        ON tp.tconst = tb.tconst
    INNER JOIN title_ratings AS tr
        ON tb.tconst = tr.tconst
    WHERE tp.category IN ('actor', 'actress')
      AND tr."numVotes" >= 1000
)
SELECT
    nconst,
    nome,
    titulo,
    votos,
    nota
FROM pessoa_titulo
WHERE rn = 1
ORDER BY votos DESC
LIMIT 100;
