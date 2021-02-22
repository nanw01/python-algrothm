/* Write your SQL below. */
select h1.name,
    h1.grade,
    h2.name,
    h2.grade,
    h3.name,
    h3.grade
from Highschooler h1,
    Highschooler h2,
    Highschooler h3,
    Likes l1,
    Likes l2
where h1.id = l1.id1
    and h2.id = l1.id2
    and(
        h2.id = l2.id1
        and h3.id = l2.id2
        and h3.id != h1.id
    )