select m1.title
from movie m1
where m1.mid not in (
        select mid
        from rating
    )