/* Write your SQL below. */
select h1.name,
    h1.grade
from Highschooler h1
where h1.grade not in (
        select h2.grade
        from Highschooler h2,
            friend f
        where h1.id = f.id1
            and h2.id = f.id2
    )