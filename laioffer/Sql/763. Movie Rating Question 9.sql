/* Write your SQL below. */
SELECT AVG(Before1980.avg) - AVG(After1980.avg)
FROM (
        SELECT AVG(stars) AS avg
        FROM Movie
            INNER JOIN Rating USING(mId)
        WHERE year < 1980
        GROUP BY mId
    ) AS Before1980,
    (
        SELECT AVG(stars) AS avg
        FROM Movie
            INNER JOIN Rating USING(mId)
        WHERE year > 1980
        GROUP BY mId
    ) AS After1980;