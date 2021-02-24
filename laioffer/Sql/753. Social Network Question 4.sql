/* Write your SQL below. */
select count(distinct f1.id1)
from friend f1,
    friend f2,
    highschooler h1,
    highschooler h2,
    highschooler h3
where f1.id1 = h3.id
    and f1.id2 = f2.id1
    and f1.id2 = h1.id
    and f2.id2 = h2.id
    and (
        h1.name = 'Cassandra'
        or (
            h1.name != 'Cassandra'
            and h2.name = 'Cassandra'
        )
    )
    and h3.name != 'Cassandra'