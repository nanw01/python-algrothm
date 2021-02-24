select re.name,
    m1.title,
    ra.stars,
    ra.ratingdate
from movie m1,
    reviewer re,
    rating ra
where m1.mid = ra.mid
    and re.rid = ra.rid
order by name,
    title,
    stars