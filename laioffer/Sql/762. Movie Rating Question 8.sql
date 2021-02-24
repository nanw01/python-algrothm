SELECT m1.title,
    max(ra1.stars) - min(ra1.stars) dif
from rating ra1
    left join movie m1 using(mid)
group by ra1.mid
order by dif desc,
    m1.title