# Write your MySQL query statement below
SELECT MAX(num) AS num
FROM (
        SELECT num
        FROM my_numbers
        GROUP BY num
        HAVING COUNT(num) = 1
    ) AS t;
# Write your MySQL query statement below
select max(t.num) as num
from (
        select m.num as num
        from my_numbers as m
        group by m.num
        having count(m.num) = 1
    ) as t