/* Write your SQL below. */
select h1.name,
    h1.grade
from highschooler h1,
    friend f1
where h1.id = f1.id1
group by 1,
    2
having count(*) = (
        select count(f2.id2) c
        from highschooler h2,
            friend f2
        where h2.id = f2.id1
        group by h2.id
        order by c desc
        limit 1
    )