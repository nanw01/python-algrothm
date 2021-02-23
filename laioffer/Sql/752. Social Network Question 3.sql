/* Write your SQL below. */
select avg(c)
from (
        select count(*) c
        from friend f
        group by f.id1
    )