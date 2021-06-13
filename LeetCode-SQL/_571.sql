# Write your MySQL query statement below
select avg(Number) median
from (
        select Number,
            sum(Frequency) over (
                order by Number asc
            ) c1,
            sum(Frequency) over (
                order by Number desc
            ) c2,
            sum(Frequency) over () cnt
        from Numbers
    ) t
where c1 >= cnt / 2
    and c2 >= cnt / 2;