/* Write your SQL below. */
SELECT m1.title,
    max(r1.stars)
FROM rating r1
    left join movie m1 on r1.mid = m1.mid
group by r1.mid
ORDER BY m1.title