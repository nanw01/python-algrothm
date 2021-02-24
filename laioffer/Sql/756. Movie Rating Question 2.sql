/* Write your SQL below. */
SELECT distinct year
from movie m1,
    rating r1
where m1.mid = r1.mid
    and r1.stars in (4, 5)
order by year