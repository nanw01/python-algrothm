/* Write your SQL below. */
-- select name,
--     title
-- from movie m
--     inner join rating r1 on m.mid = r1.mid
--     inner join rating r2 on r1.rid = r2.rid
--     inner join reviewer re on r1.rid = re.rid
-- where r1.mid = r2.mid
--     and r1.ratingdate < r2.ratingDate
--     and r1.stars < r2.stars
--
--
--
--
--
SELECT re1.name,
    m1.title
from movie m1,
    reviewer re1,
    reviewer re2,
    rating ra1,
    rating ra2
where re1.rid = ra1.rid
    and re2.rid = ra2.rid
    and re1.rid = re2.rid
    and ra1.stars < ra2.stars
    and ra1.ratingDate < ra2.ratingDate
    and m1.mid = ra1.mid
    and ra1.mid = ra2.mid