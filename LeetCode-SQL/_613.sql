# Write your MySQL query statement below
select min(abs(p1.x - p2.x)) as shortest
from point as p1
    join point as p2
where p1.x <> p2.x